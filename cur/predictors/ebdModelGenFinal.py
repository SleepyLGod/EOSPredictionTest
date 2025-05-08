import os
import math
# import json # 未直接使用
import bisect
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import sys # For exiting
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast

# os.environ['NCCL_DEBUG'] = 'INFO' # set NCCL debug level

# torch.backends.cuda.max_split_size_mb = 128 # set max split size to avoid fragmentation
# torch.cuda.set_per_process_memory_fraction(0.9) # limit GPU memory usage

HIDDEN_SIZE = 8192
BATCH_SIZE_PER_GPU = 32
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-3
EPOCHS = 50
USE_AMP = True
MEMORY_FRACTION_PER_GPU = 0.8
MAX_CONSECUTIVE_EMPTY_BATCHES = 20 # 如果连续这么多批次为空，则终止
EMPTY_BATCH_EPOCH_THRESHOLD = 0.5 # 如果一个epoch中空批次比例超过此值，发出严重警告
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUR_DIR = Path(__file__).parent.absolute() if "__file__" in locals() else Path.cwd()
FEATURE_DIR = CUR_DIR.parent / "training_data" / "ebd" / "features" / "llama3_70b"
if not FEATURE_DIR.exists():
    raise FileNotFoundError(f"feature directory non-exist: {FEATURE_DIR}")
SAVE_DIR_BASE = CUR_DIR / "saved_models"
LOG_DIR_BASE = CUR_DIR / "logs"
# SAVE_DIR.mkdir(exist_ok=True)
# model_save_path = SAVE_DIR / "enhanced_mlp.pth"

# LOG_DIR.mkdir(exist_ok=True)
# log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.FileHandler(log_file), logging.StreamHandler()] # log to both file and console
# )

def setup_logging(rank, log_dir_base):
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_log_dir = log_dir_base / run_timestamp
    current_log_dir.mkdir(parents=True, exist_ok=True) # exist_ok=True handles potential (though unlikely) race conditions

    log_file_suffix = f"_rank{rank}" if dist.is_initialized() and dist.get_world_size() > 1 else ""
    log_file = current_log_dir / f"train{log_file_suffix}.log"
    
    handlers = [logging.FileHandler(log_file)]
    is_main_process = rank == 0
    
    if is_main_process:
        handlers.append(logging.StreamHandler())
        
    log_format = "%(asctime)s [%(levelname)s]"
    if dist.is_initialized() and dist.get_world_size() > 1:
        log_format += " (Rank %(rank)s)"
    log_format += " %(message)s"

    root_logger = logging.getLogger()
    # Clear existing handlers from root logger to avoid duplicate logs if script is re-run in same session
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    formatter = logging.Formatter(log_format)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    if dist.is_initialized() and dist.get_world_size() > 1:
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.rank = dist.get_rank()
            return record
        if logging.getLogRecordFactory() != record_factory: # Avoid re-setting if already set
            logging.setLogRecordFactory(record_factory)
    
    return current_log_dir

def get_current_save_dir(save_dir_base, log_dir):
    run_timestamp = log_dir.name
    current_save_dir = save_dir_base / run_timestamp
    current_save_dir.mkdir(parents=True, exist_ok=True)
    return current_save_dir


def decode_params(encoded):
    encoded = int(encoded)
    return {
        'temperature': ((encoded >> 24) & 0xFF) / 255 * 0.8 + 0.1,
        'top_k': (encoded >> 17) & 0x7F,
        'repetition_penalty': ((encoded >> 9) & 0xFF) / 255 * 0.3 + 1.3,
        'max_new_tokens': ((encoded >> 6) & 0x7) * 100
    }

class EmbeddingDataset(Dataset): # Using the robust version from previous iteration
    def __init__(self, feature_dir, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.file_paths = []
        self.cumulative_samples = []
        self.file_sample_counts = []
        
        if not feature_dir.exists():
            logging.critical(f"Feature directory non-exist: {feature_dir}")
            raise FileNotFoundError(f"Feature directory non-exist: {feature_dir}")

        sorted_files = sorted(os.listdir(feature_dir))
        for f_name in sorted_files:
            file_path = os.path.join(feature_dir, f_name)
            if not file_path.endswith('.npz'):
                continue
            try:
                with np.load(file_path, allow_pickle=True) as data: 
                    if not ('features' in data and 'labels' in data):
                        logging.warning(f"File {file_path} missing 'features' or 'labels' array. Skipping.")
                        continue
                    features, labels = data['features'], data['labels']
                    
                    if not (isinstance(features, np.ndarray) and isinstance(labels, np.ndarray)):
                        logging.warning(f"File {file_path}: 'features' or 'labels' are not numpy arrays. Skipping.")
                        continue
                    if features.ndim == 0 or labels.ndim == 0: 
                        features = np.atleast_1d(features)
                        labels = np.atleast_1d(labels)

                    if len(features) == 0: 
                        logging.warning(f"File {file_path} contains empty 'features' array. Skipping.")
                        continue
                    if len(features) != len(labels):
                        logging.warning(f"File {file_path} has length mismatch between features ({len(features)}) and labels ({len(labels)}). Skipping.")
                        continue
                    if not (features.dtype == object and labels.dtype == object):
                        logging.warning(f"File {file_path}: Expected object dtype for features/labels, got {features.dtype}/{labels.dtype}. Skipping.")
                        continue
                    if len(features) > 0 and not (isinstance(features[0], dict) and isinstance(labels[0], dict)): # Check first element if not empty
                        logging.warning(f"File {file_path}: First element of features/labels is not a dict. Skipping.")
                        continue
                        
                    n_samples = len(features)
                    self.file_paths.append(file_path)
                    self.file_sample_counts.append(n_samples)
            except Exception as e:
                logging.error(f"Corrupted or invalid NPZ file {file_path}: {str(e)}. Skipping.")
                continue
        
        if not self.file_paths:
            logging.critical(f"No valid .npz files found in dataset directory {feature_dir}")
            raise RuntimeError(f"Dataset directory {feature_dir} contains no valid .npz files")
        
        # calculate cumulative samples
        total = 0
        for n_samples_in_file in self.file_sample_counts: # Renamed for clarity
            self.cumulative_samples.append((total, total + n_samples_in_file))
            total += n_samples_in_file
        self.total_samples = total
        if self.rank == 0:
            logging.info(f"Initialized dataset. Total valid files: {len(self.file_paths)}, Total samples: {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    # def _normalize_value(self, value, min_val, max_val, name):
    #     if max_val == min_val:
    #         norm_val = 0.0 if value == min_val else 0.5 
    #         if self.rank == 0 and random.random() < 0.001 : # Log very sparsely for this case
    #             logging.debug(f"Normalization for {name}: min=max={min_val}, value={value}, norm_val={norm_val}")
    #     else:
    #         norm_val = (value - min_val) / (max_val - min_val)
        
    #     clipped_val = np.clip(norm_val, 0.0, 1.0)
    #     if clipped_val != norm_val and abs(clipped_val - norm_val) > 1e-6 : 
    #         if self.rank == 0 and random.random() < 0.01 : # Log 1% of clipping events from rank 0
    #             logging.warning(f"Value for {name} ({value}) was clipped after normalization. Original norm: {norm_val:.4f}, clipped: {clipped_val:.4f}. Check MIN/MAX constants for {name}.")
    #     return np.float32(clipped_val)

    def __getitem__(self, idx):
        if not (0 <= idx < self.total_samples):
            # This log can be very verbose if sampler generates out-of-bound idx temporarily before epoch sync
            logging.error(f"Index {idx} out of bounds (0, {self.total_samples-1})") 
            return None 
        
        # get the file index and sample index
        start_list = [cs[0] for cs in self.cumulative_samples]
        file_idx = bisect.bisect_right(start_list, idx) - 1
        
        if not (0 <= file_idx < len(self.file_paths)):
            logging.error(f"Calculated file_idx {file_idx} is out of bounds for idx {idx}")
            return None
            
        start, _ = self.cumulative_samples[file_idx]
        sample_idx_in_file = idx - start
        
        if not (0 <= sample_idx_in_file < self.file_sample_counts[file_idx]):
            logging.error(f"Sample index {sample_idx_in_file} out of bounds for file {self.file_paths[file_idx]} (size {self.file_sample_counts[file_idx]})")
            return None
            
        try:
            with np.load(self.file_paths[file_idx], allow_pickle=True, mmap_mode='r') as data:
                feat_dict = data['features'][sample_idx_in_file]
                lab_dict = data['labels'][sample_idx_in_file]
                
                required_feat_keys = ['sys_para', 'pos', 'ebd']
                required_lab_keys = ['rest_len']
                for key in required_feat_keys:
                    if key not in feat_dict:
                        raise KeyError(f"Missing key '{key}' in features dict (file {self.file_paths[file_idx]}, sample {sample_idx_in_file})")
                for key in required_lab_keys:
                    if key not in lab_dict:
                        raise KeyError(f"Missing key '{key}' in labels dict (file {self.file_paths[file_idx]}, sample {sample_idx_in_file})")
                
                encoded_param = feat_dict['sys_para']
                seq_pos = feat_dict['pos'] # This is the raw sequence position
                embedding = feat_dict['ebd'] 
                rest_len = lab_dict['rest_len']

                if not isinstance(embedding, np.ndarray) or embedding.shape != (HIDDEN_SIZE,):
                    raise ValueError(f"Embedding has incorrect type or shape. Expected ({HIDDEN_SIZE},), got {type(embedding)} with shape {embedding.shape if isinstance(embedding, np.ndarray) else 'N/A'}")

                params = decode_params(encoded_param)
                
                # return {
                #     'temperature': self._normalize_value(params['temperature'], MIN_TEMP, MAX_TEMP, 'temperature'),
                #     'top_k': self._normalize_value(params['top_k'], MIN_TOP_K, MAX_TOP_K, 'top_k'),
                #     'repetition_penalty': self._normalize_value(params['repetition_penalty'], MIN_REP_PENALTY, MAX_REP_PENALTY, 'repetition_penalty'),
                #     'max_len': self._normalize_value(params['max_new_tokens'], MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS, 'max_new_tokens'),
                #     'seq_pos': self._normalize_value(float(seq_pos), MIN_SEQ_POS, MAX_SEQ_POS, 'seq_pos'), # Ensure seq_pos is float for normalization
                #     'embedding': embedding.astype(np.float32),
                #     'remaining': np.float32(rest_len),
                # }
                return {
                    'temperature': np.float32((params['temperature'] - 0.1) / 0.8),
                    'top_k': np.float32(params['top_k'] / 100.0),
                    'repetition_penalty': np.float32((params['repetition_penalty'] - 1.3) / 0.3),
                    'max_len': np.float32((params['max_new_tokens'] - 300) / 200.0),  
                    'seq_pos': np.float32(seq_pos / 4096.0),
                    'embedding': embedding.astype(np.float32),
                    'remaining': np.float32(rest_len),
                    # 'over_max': np.float32(over_max)
                }
        except Exception as e:
            # Log less verbosely to avoid flooding logs, maybe sample error logging
            if random.random() < 0.05: # Log 5% of these errors
                logging.error(f"Error processing sample: File {self.file_paths[file_idx]}, sample_idx_in_file {sample_idx_in_file}. Error: {str(e)}")
            return None

class EnhancedMLP(nn.Module): 
    def __init__(self, hidden_size):
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Linear(hidden_size + 5, 2048), 
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.2), 
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
        self.residual = nn.Linear(hidden_size + 5, 1024)
    
    def forward(self, x_dict):
        params_to_cat = [x_dict['embedding']]
        for param_name in ['temperature', 'top_k', 'repetition_penalty', 'max_len', 'seq_pos']:
            tensor = x_dict[param_name]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            params_to_cat.append(tensor)
            
        main_feature = torch.cat(params_to_cat, dim=1)
        interacted = self.interaction(main_feature)
        residual_out = self.residual(main_feature)
        fused = interacted + residual_out
        reg_out = self.reg_head(fused).squeeze(-1)
        return reg_out

def collate_fn(batch_list): 
    original_size = len(batch_list)
    valid_samples = [s for s in batch_list if isinstance(s, dict) and s] 
    
    num_filtered = original_size - len(valid_samples)
    if num_filtered > 0:
        if random.random() < 0.01 and dist.is_initialized() and dist.get_rank() == 0 : 
            logging.warning(f"Collate (Rank 0 sample): Filtered {num_filtered}/{original_size} invalid samples from a batch.")
        # This specific high failure rate check is probably better done per epoch or via consecutive empty batches logic
        if original_size > 0 and (num_filtered / original_size) > 0.8: 
            logging.critical(f"Collate: Extremely high failure rate! Filtered {num_filtered}/{original_size} samples.")
    
    if not valid_samples:
        return {} 
    
    try:
        collated_batch = torch.utils.data.dataloader.default_collate(valid_samples)
    except RuntimeError as e: 
        if dist.is_initialized() and dist.get_rank() == 0:
            logging.error(f"Error in default_collate (Rank 0): {e}. Samples causing issues (first few): {valid_samples[:2]}")
        return {} 
    except Exception as e: 
        if dist.is_initialized() and dist.get_rank() == 0:
            logging.error(f"Unexpected error in default_collate (Rank 0): {e}. Samples: {valid_samples[:2]}")
        return {}
    
    if 'embedding' not in collated_batch or not torch.isfinite(collated_batch['embedding']).all():
        logging.error("Collate: NaN/Inf in 'embedding' or key missing.") # Can be too verbose
        return {}
        
    for key in ['temperature', 'top_k', 'repetition_penalty', 'max_len', 'seq_pos', 'remaining']:
        if key not in collated_batch:
            logging.error(f"Collate: Key '{key}' missing in collated batch.")
            return {}
        if not torch.isfinite(collated_batch[key]).all():
            logging.error(f"Collate: NaN/Inf in '{key}'.")
            return {}
    for key in ['temperature', 'top_k', 'repetition_penalty', 'max_len', 'seq_pos']:
        if collated_batch[key].min() < (0.0 - 1e-6) or collated_batch[key].max() > (1.0 + 1e-6):
            if random.random() < 0.01 and dist.is_initialized() and dist.get_rank() == 0 :
                logging.error(f"Collate (Rank 0 sample): Invalid value range in '{key}': min={collated_batch[key].min():.3f}, max={collated_batch[key].max():.3f}. Expected [0,1].")
            # Potentially return {} if strict range adherence is critical, but clipping in dataset should handle major issues
            
    return collated_batch

def setup_ddp(rank, world_size, memory_fraction):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_timeout = timedelta(minutes=20)
    dist.init_process_group(
        backend="nccl",
        # init_method='env://', # 或者保持默认，如果 MASTER_ADDR/PORT 已设置
        rank=rank,
        world_size=world_size,
        timeout=init_timeout
    )
    
    torch.cuda.set_device(rank)
    if memory_fraction > 0 and memory_fraction <= 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=rank)
            if rank == 0: logging.info(f"Set per-process memory fraction to {memory_fraction} for GPU {rank}")
        except AttributeError: # MODIFICATION: Catch AttributeError for older PyTorch
            if rank == 0: logging.warning(
                f"torch.cuda.set_per_process_memory_fraction not available in this PyTorch version. Memory limit not set for GPU {rank}."
            )
        except Exception as e: # Catch other potential errors
            if rank == 0: logging.warning(f"Failed to set memory fraction for GPU {rank}: {e}")

def cleanup_ddp():
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    current_log_dir = setup_logging(rank, args['log_dir_base'])
    current_save_dir = get_current_save_dir(args['save_dir_base'], current_log_dir)

    # --- 1. 初始化 DDP 环境 (包括 init_process_group) ---
    if world_size > 1:
        setup_ddp(rank, world_size, args['memory_fraction'])
    
    device = rank # rank is the GPU ID for this process in DDP

    # --- 2. Huber Loss Delta Calculation & Synchronization ---
    # 创建一个 tensor 来持有 delta 值，所有进程都创建它。Rank 0 会更新它。
    # 使用 .to(device) 确保 tensor 在正确的 GPU 上，以便 NCCL broadcast
    huber_delta_tensor = torch.tensor(1.0, dtype=torch.float32, device=device) # Default value

    if rank == 0:
        logging.info("Rank 0: Preparing to calculate HuberLoss delta...")
        rest_lengths_for_delta_calc = []
        actual_samples_for_delta_calc = 0
        
        # Rank 0 进行数据扫描以收集 rest_len
        # 注意: 为了计算 delta，Rank 0 可能需要访问整个数据集的元信息或样本。
        # 这里的 EmbeddingDataset 实例是为了统计，不是 DDP 训练用的那个。
        # 如果这个初始化本身很慢，可以考虑优化这部分数据的收集方式。
        try:
            # 临时的、仅 Rank 0 使用的数据集实例，用于统计
            # 传递 rank=0, world_size=1 确保它看到所有文件（如果其内部逻辑支持）
            # 或者，如果 EmbeddingDataset 的默认行为就是加载所有，则不需要 rank/world_size 参数
            temp_stat_dataset = EmbeddingDataset(args['feature_dir']) # rank=0, world_size=1 (或者不传，取决于实现)
            
            if len(temp_stat_dataset) > 0:
                # 考虑减少样本量以加快速度，例如10万或20万，而不是50万
                sample_size_for_delta = min(len(temp_stat_dataset), args.get('delta_calc_sample_size', 200000)) 
                indices = random.sample(range(len(temp_stat_dataset)), sample_size_for_delta)
                actual_samples_for_delta_calc = len(indices)
                
                logging.info(f"Rank 0: Scanning {actual_samples_for_delta_calc} random samples for Huber delta calculation...")
                for i in tqdm(indices, desc="Scanning rest_len for Huber delta", disable=(rank!=0)): # tqdm 只在 rank 0 显示
                    sample = temp_stat_dataset[i]
                    if sample and 'remaining' in sample and np.isfinite(sample['remaining']):
                        rest_lengths_for_delta_calc.append(sample['remaining'])
            else:
                logging.warning("Rank 0: Temporary dataset for delta calculation is empty.")

            del temp_stat_dataset # 及时释放资源
        except Exception as e:
            logging.error(f"Rank 0: Error during data collection for Huber delta: {e}. Using default delta.")
            rest_lengths_for_delta_calc = [] #确保列表为空，使用默认delta

        logging.info(f"Rank 0: Collected {len(rest_lengths_for_delta_calc)} valid 'rest_len' values from {actual_samples_for_delta_calc} scanned samples.")

        # Rank 0 计算 delta
        calculated_huber_delta = 1.0 # Default
        if rest_lengths_for_delta_calc:
            try:
                rest_arr = np.array(rest_lengths_for_delta_calc, dtype=np.float32)
                if len(rest_arr) == 0: raise ValueError("Empty array after collecting finite rest_lengths for delta.")
                
                rest_mean = np.nanmean(rest_arr)
                if np.isnan(rest_mean): raise ValueError("Mean of rest_lengths is NaN for delta calculation.")
                
                abs_diff = np.abs(rest_arr - rest_mean)
                c_delta = np.nanquantile(abs_diff, 0.9) # 0.9 quantile of absolute differences
                
                if not (np.isnan(c_delta) or c_delta <= 1e-6): # Ensure delta is a sensible positive value
                    calculated_huber_delta = float(c_delta)
                else:
                    logging.warning(f"Rank 0: Calculated delta was problematic ({c_delta}). Using default delta 1.0.")
            except Exception as e:
                logging.warning(f"Rank 0: Failed to calculate HuberLoss delta from collected data (Reason: {str(e)}). Using default delta 1.0.")
        
        huber_delta_tensor[0] = calculated_huber_delta # 更新 Rank 0 上的 tensor 值
        logging.info(f"Rank 0: Calculated Huber delta: {huber_delta_tensor.item():.4f}")

    # --- 3. 同步 Huber Delta 值 ---
    if world_size > 1:
        # 所有进程参与广播，将 Rank 0 的 huber_delta_tensor 值广播给所有其他进程
        dist.broadcast(huber_delta_tensor, src=0)
    
    # 所有进程从其本地的 (可能已更新的) tensor 中获取 float 类型的 delta 值
    huber_delta = huber_delta_tensor.item() 
    # 确保所有rank都记录它们将使用的delta
    logging.info(f"Rank {rank}: Using HuberLoss with delta: {huber_delta:.4f}")


    # --- 4. 初始化用于训练的数据集和 DataLoader (DDP 感知) ---
    dataset = EmbeddingDataset(args['feature_dir'], rank=rank, world_size=world_size)
    
    sampler = None
    dataloader_shuffle = True
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        dataloader_shuffle = False

    dataloader = DataLoader(
        dataset, 
        batch_size=args['batch_size_per_gpu'], 
        shuffle=dataloader_shuffle,
        num_workers=args['num_workers'], 
        pin_memory=True, 
        persistent_workers=True if args['num_workers'] > 0 else False,
        collate_fn=collate_fn, 
        sampler=sampler, 
        drop_last=True # drop_last=True is important for DDP
    )
    
    # --- 5. 模型、优化器、调度器、损失函数初始化 ---
    model = EnhancedMLP(args['hidden_size']).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False) 
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args['learning_rate'], 
        weight_decay=1e-4, 
        betas=(0.9, 0.999)
    )
    
    # 计算调度器的总步数
    # len(dataloader) 是每个 GPU 的批次数，这对于梯度累积是正确的
    num_update_steps_per_epoch = len(dataloader) // args['gradient_accumulation_steps']
    if len(dataloader) % args['gradient_accumulation_steps'] != 0:
        num_update_steps_per_epoch +=1 
    
    total_scheduler_steps = args['epochs'] * num_update_steps_per_epoch
    if rank == 0: 
        logging.info(f"DataLoader length (batches per GPU per epoch): {len(dataloader)}")
        logging.info(f"Optimizer update steps per epoch per GPU: {num_update_steps_per_epoch}")
        logging.info(f"Total scheduler steps for OneCycleLR: {total_scheduler_steps}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args['learning_rate'], 
        total_steps=total_scheduler_steps,
        pct_start=0.3, 
        div_factor=10, 
        final_div_factor=100
    )
    
    reg_criterion = nn.HuberLoss(delta=huber_delta) # 使用同步后的 delta
    scaler = GradScaler(enabled=args['use_amp'])
    
    # --- 6. 训练循环 ---
    best_loss = float('inf')
    model_save_path_best = current_save_dir / "enhanced_mlp_best.pth"
    consecutive_empty_batches = 0
    
    for epoch in range(args['epochs']):
        if sampler: sampler.set_epoch(epoch) # DDP sampler需要设置 epoch
        model.train()
        total_loss_epoch_accum = 0.0    # 用于记录当前rank的累积损失 (未除以累积步数)
        num_optimizer_steps_epoch = 0 # 当前rank实际执行的优化器步数
        total_processed_micro_batches = 0 # 处理的微批次数
        total_empty_batches_in_epoch = 0
        
        data_iterator = dataloader
        if rank == 0:
            data_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args['epochs']}", unit="batch")
        
        optimizer.zero_grad() # 在每个epoch或累积周期开始时清零梯度
        
        for batch_idx, batch_data in enumerate(data_iterator):
            if not batch_data: # 如果 collate_fn 返回空字典
                total_empty_batches_in_epoch += 1
                consecutive_empty_batches += 1
                if rank == 0: 
                    logging.warning(f"Skipping empty batch at epoch {epoch+1}, micro_batch_idx {batch_idx}. Consecutive empty: {consecutive_empty_batches}")
                if consecutive_empty_batches >= args['max_consecutive_empty_batches']:
                    if rank == 0:
                        logging.critical(f"Stopping training: Exceeded maximum consecutive empty batches ({args['max_consecutive_empty_batches']}).")
                    if world_size > 1: dist.barrier() # 确保所有进程在退出前有机会记录
                    sys.exit(f"Rank {rank} exiting due to too many consecutive empty batches.")
                continue
            
            consecutive_empty_batches = 0 # 重置计数器
            total_processed_micro_batches +=1
            
            try:
                input_features = {k: v.to(device, non_blocking=True) for k, v in batch_data.items() if k != 'remaining'}
                reg_labels = batch_data['remaining'].to(device, non_blocking=True)
            except Exception as e:
                if rank == 0: logging.error(f"Error moving batch {batch_idx} to device: {e}. Skipping batch.")
                continue # 跳过这个有问题的批次
            
            with autocast(enabled=args['use_amp']):
                reg_pred = model(input_features)
                loss = reg_criterion(reg_pred, reg_labels)
            
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                if rank == 0: logging.warning(f"NaN/Inf loss detected at epoch {epoch+1}, micro_batch {batch_idx}. Skipping gradient update for this batch.")
                # 梯度已经是NaN/Inf，不需要累积，但要确保优化器不会在坏梯度上更新
                # 如果这是累积周期的第一步，后续的累积也可能受影响。
                # 这里选择跳过这个微批次的梯度贡献。
                continue 
            
            loss_val_for_reporting = loss.item() # 用于累加报告的损失值 (未除以累积步数)
            
            # 为梯度累积缩放损失
            scaled_loss = loss / args['gradient_accumulation_steps']
            
            scaler.scale(scaled_loss).backward() # 反向传播缩放后的损失
            
            total_loss_epoch_accum += loss_val_for_reporting # 累加原始损失值
            
            # 执行优化器步骤和梯度清零
            if (batch_idx + 1) % args['gradient_accumulation_steps'] == 0 or \
                (batch_idx + 1) == len(dataloader): # 在累积了足够步数或到达 epoch 末尾时
                
                scaler.unscale_(optimizer) # 在裁剪前 unscale 梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
                
                scaler.step(optimizer) # 执行优化器步骤 (如果梯度有效)
                scaler.update() # 更新 GradScaler 的缩放因子
                
                optimizer.zero_grad() # 清零梯度，为下一个累积周期或下一个 epoch 做准备
                scheduler.step() # 在优化器步骤之后更新学习率
                num_optimizer_steps_epoch +=1
        
        # 计算当前 rank 的平均损失 (基于处理过的微批次)
        avg_loss_this_rank = total_loss_epoch_accum / total_processed_micro_batches if total_processed_micro_batches > 0 else 0.0
        
        # 同步所有 rank 的平均损失以获得全局平均损失
        if world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss_this_rank, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG) # 计算所有rank的平均损失
            avg_loss_epoch_global = avg_loss_tensor.item()
        else:
            avg_loss_epoch_global = avg_loss_this_rank
        
        current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
        
        if rank == 0:
            logging.info(
                f"Epoch {epoch+1} Summary: Avg Global Loss (per micro-batch): {avg_loss_epoch_global:.4f} | "
                f"LR: {current_lr:.2e} | OptSteps This Epoch: {num_optimizer_steps_epoch}"
            )
            empty_batch_ratio = total_empty_batches_in_epoch / len(dataloader) if len(dataloader) > 0 else 0
            logging.info(f"Epoch {epoch+1}: Empty micro-batch ratio: {empty_batch_ratio:.2%} ({total_empty_batches_in_epoch}/{len(dataloader)})")
            if empty_batch_ratio > args['empty_batch_epoch_threshold']:
                logging.critical(
                    f"Epoch {epoch+1}: High empty micro-batch ratio ({empty_batch_ratio:.2%}) exceeded threshold "
                    f"({args['empty_batch_epoch_threshold']:.0%}). Consider data quality check."
                )
            
            if avg_loss_epoch_global < best_loss and total_processed_micro_batches > 0 : # 只有在处理了数据且损失更好时才保存
                best_loss = avg_loss_epoch_global
                logging.info(f"Best global loss updated: {best_loss:.4f}. Saving model to {model_save_path_best}")
                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(state_to_save, model_save_path_best)
            
            # 保存检查点
            checkpoint_path = current_save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss_epoch_global, # 保存的是全局平均损失
                'best_loss': best_loss,
                'huber_delta': huber_delta, # 保存实际使用的 delta
                'args': args # 保存用于此次训练的参数
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    # --- 7. 训练结束后的清理 ---
    if world_size > 1:
        cleanup_ddp()
    if rank == 0:
        logging.info(f"Training complete. Best model saved at {model_save_path_best}")
        logging.info(f"Final logs and checkpoints are in {current_log_dir.parent}") # 指向包含时间戳的父目录

def datacheck():
    problem_file = "meta-llama_Meta-Llama-3-70B_t0.9_tk100_r1.3_mok300_qa3464.npz"
    path = os.path.join(FEATURE_DIR, problem_file)
    with np.load(path, allow_pickle=True) as data:
            print("Features dtype:", data['features'].dtype)
            print("First feature:", data['features'][0])
            print("Labels dtype:", data['labels'].dtype)
            print("First label:", data['labels'][0])
            print()

def check_dataset(feature_dir):
    for f in os.listdir(feature_dir):
        if not f.endswith('.npz'):
            continue
        file_path = os.path.join(feature_dir, f)
        try:
            with np.load(file_path, allow_pickle=True) as data:
                features = data['features']
                labels = data['labels']
                logging.info(f"{file_path}: {len(features)} samples")
                assert len(features) == len(labels), "Length mismatch"
        except Exception as e:
            logging.error(f"Error in {file_path}: {str(e)}")

def inspect_problem_file():
    problem_file = "meta-llama_Meta-Llama-3-70B_t0.1_tk100_r1.6_mok500_qa50087.npz"
    path = os.path.join(FEATURE_DIR, problem_file)
    with np.load(path, allow_pickle=True) as data:
        features = data['features']
        labels = data['labels']
        print(f"File: {path}")
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        # check all the keys in features and labels
        for i in range(len(features)):
            try:
                feat = features[i]
                lab = labels[i]
                assert all(k in feat for k in ['sys_para', 'pos', 'ebd'])
                assert all(k in lab for k in ['rest_len', 'over_max_len'])
            except Exception as e:
                print(f"Invalid sample {i}: {str(e)}")

def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA GPUs available. Exiting.")
        return
    print(f"Found {world_size} GPUs. Starting DDP training...")
    
    args = {
        'feature_dir': FEATURE_DIR,
        'log_dir_base': LOG_DIR_BASE,
        'save_dir_base': SAVE_DIR_BASE,
        'hidden_size': HIDDEN_SIZE,
        'batch_size_per_gpu': BATCH_SIZE_PER_GPU,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'num_workers': 4,
        'use_amp': USE_AMP,
        'memory_fraction': MEMORY_FRACTION_PER_GPU,
        # Constants for empty batch checks
        'max_consecutive_empty_batches': MAX_CONSECUTIVE_EMPTY_BATCHES,
        'empty_batch_epoch_threshold': EMPTY_BATCH_EPOCH_THRESHOLD,
    }
    
    LOG_DIR_BASE.mkdir(parents=True, exist_ok=True)
    SAVE_DIR_BASE.mkdir(parents=True, exist_ok=True)
    
    mp.spawn(train_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    datacheck()
    inspect_problem_file()
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    torch.cuda.empty_cache()
    # check_dataset(FEATURE_DIR)
    logging.info(f"Available GPUs: {torch.cuda.device_count()}")
    # logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # This typically means it's already been set, or 'spawn' is not supported/default on the system.
        # Forcing can be problematic on some systems if not the first call.
        # Default 'fork' on Linux is usually fine for DDP with CUDA if globals are handled carefully.
        # 'spawn' is generally safer cross-platform and for CUDA.
        print("Note: Multiprocessing start method 'spawn' was already set or could not be forced. Continuing.")
        pass 
    main()