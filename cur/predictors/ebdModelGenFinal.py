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
from datetime import datetime
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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
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
    
    if world_size > 1:
        setup_ddp(rank, world_size, args['memory_fraction'])
    
    device = rank
    
    dataset = EmbeddingDataset(args['feature_dir'], rank=rank, world_size=world_size)
    
    rest_lengths_for_delta = []
    actual_samples_for_delta_calc = 0 # MODIFICATION: Track samples used for delta
    if rank == 0:
        logging.info("Rank 0: Scanning dataset for HuberLoss delta calculation...")
        temp_stat_dataset = EmbeddingDataset(args['feature_dir']) 
        if len(temp_stat_dataset) > 0:
            sample_size_for_delta = min(len(temp_stat_dataset), 500000)
            indices = random.sample(range(len(temp_stat_dataset)), sample_size_for_delta)
            actual_samples_for_delta_calc = len(indices) # MODIFICATION
            for i in tqdm(indices, desc="Scanning rest_len for Huber delta", disable=(rank!=0)):
                sample = temp_stat_dataset[i]
                if sample and 'remaining' in sample:
                    if np.isfinite(sample['remaining']):
                        rest_lengths_for_delta.append(sample['remaining'])
        del temp_stat_dataset
        logging.info(f"Rank 0: Used {actual_samples_for_delta_calc} samples to gather 'rest_len' for Huber delta calculation.") # MODIFICATION
    
    if world_size > 1:
        if rank == 0:
            rest_lengths_tensor = torch.tensor(rest_lengths_for_delta, dtype=torch.float32).cuda(rank)
            size_tensor = torch.tensor(len(rest_lengths_tensor), dtype=torch.long).cuda(rank)
        else: 
            size_tensor = torch.empty(1, dtype=torch.long).cuda(rank)
        dist.broadcast(size_tensor, src=0)
        if rank != 0: 
            rest_lengths_tensor = torch.empty(size_tensor.item(), dtype=torch.float32).cuda(rank)
        dist.broadcast(rest_lengths_tensor, src=0)
        rest_lengths_for_delta = rest_lengths_tensor.cpu().tolist()
    
    huber_delta = 1.0 
    if not rest_lengths_for_delta:
        if rank == 0: logging.warning("No valid 'rest_len' data found for HuberLoss delta. Using default delta=1.0.")
    else:
        try:
            rest_arr = np.array(rest_lengths_for_delta, dtype=np.float32)
            if len(rest_arr) == 0: raise ValueError("Empty array after collecting finite rest_lengths")
            rest_mean = np.nanmean(rest_arr)
            if np.isnan(rest_mean): raise ValueError("Mean of rest_lengths is NaN.")
            abs_diff = np.abs(rest_arr - rest_mean)
            calculated_delta = np.nanquantile(abs_diff, 0.9)
            if np.isnan(calculated_delta) or calculated_delta <= 1e-6:
                if rank == 0: logging.warning(f"Calculated delta is problematic ({calculated_delta}), using default delta=1.0.")
            else:
                huber_delta = float(calculated_delta)
        except Exception as e:
            if rank == 0: logging.warning(f"Failed to calculate HuberLoss delta (Reason: {str(e)}). Using default delta=1.0.")
    if rank == 0: logging.info(f"Using HuberLoss with delta: {huber_delta:.4f}")
    
    sampler = None
    dataloader_shuffle = True
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        dataloader_shuffle = False
    
    dataloader = DataLoader(
        dataset, batch_size=args['batch_size_per_gpu'], shuffle=dataloader_shuffle,
        num_workers=args['num_workers'], pin_memory=True, 
        persistent_workers=True if args['num_workers'] > 0 else False,
        collate_fn=collate_fn, sampler=sampler, drop_last=True
    )
    
    model = EnhancedMLP(args['hidden_size']).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=False) 
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args['learning_rate'], weight_decay=1e-4, betas=(0.9, 0.999)
    )
    
    num_update_steps_per_epoch = len(dataloader) // args['gradient_accumulation_steps']
    if len(dataloader) % args['gradient_accumulation_steps'] != 0:
        num_update_steps_per_epoch +=1
    total_scheduler_steps = args['epochs'] * num_update_steps_per_epoch
    if rank == 0: 
        logging.info(f"Effective batches per epoch per GPU (len(dataloader)): {len(dataloader)}")
        logging.info(f"Optimizer update steps per epoch: {num_update_steps_per_epoch}")
        logging.info(f"Total scheduler steps for OneCycleLR: {total_scheduler_steps}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args['learning_rate'], total_steps=total_scheduler_steps,
        pct_start=0.3, div_factor=10, final_div_factor=100
    )
    
    reg_criterion = nn.HuberLoss(delta=huber_delta)
    scaler = GradScaler(enabled=args['use_amp'])
    
    best_loss = float('inf')
    model_save_path_best = current_save_dir / "enhanced_mlp_best.pth"
    
    # MODIFICATION: Counters for empty batch monitoring
    consecutive_empty_batches = 0
    
    for epoch in range(args['epochs']):
        if sampler: sampler.set_epoch(epoch)
        model.train()
        total_loss_epoch_accum = 0.0
        num_optimizer_steps_epoch = 0
        total_empty_batches_in_epoch = 0 # MODIFICATION
        data_iterator = dataloader
        if rank == 0:
            data_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args['epochs']}", unit="batch")
        
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(data_iterator):
            if not batch_data: 
                total_empty_batches_in_epoch += 1 # MODIFICATION
                consecutive_empty_batches += 1    # MODIFICATION
                if rank == 0: 
                    logging.warning(f"Skipping empty batch at epoch {epoch+1}, raw_batch_idx {batch_idx}. Consecutive empty: {consecutive_empty_batches}")
                if consecutive_empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES: # MODIFICATION
                    if rank == 0:
                        logging.critical(f"Stopping training: Exceeded maximum consecutive empty batches ({MAX_CONSECUTIVE_EMPTY_BATCHES}). Check data quality or dataset/collate_fn logic.")
                    if world_size > 1: dist.barrier() # Ensure all processes log before exiting
                    sys.exit(f"Rank {rank} exiting due to too many consecutive empty batches.") # Exit all processes
                continue
            
            consecutive_empty_batches = 0 # MODIFICATION: Reset if batch is valid
            
            try:
                input_features = {k: v.to(device, non_blocking=True) for k, v in batch_data.items() if k != 'remaining'}
                reg_labels = batch_data['remaining'].to(device, non_blocking=True)
            except Exception as e:
                if rank == 0: logging.error(f"Error moving batch to device: {e}. Skipping batch.")
                continue
            
            with autocast(enabled=args['use_amp']):
                reg_pred = model(input_features)
                loss = reg_criterion(reg_pred, reg_labels)
            
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                if rank == 0: logging.warning(f"NaN/Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping grad update for this batch.")
                continue 
            
            loss_val_for_accum = loss.item() # Store for logging before scaling for grad accum
            loss = loss / args['gradient_accumulation_steps']
            scaler.scale(loss).backward()
            total_loss_epoch_accum += loss_val_for_accum # Accumulate original loss value for averaging
            
            if (batch_idx + 1) % args['gradient_accumulation_steps'] == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                num_optimizer_steps_epoch +=1
        
        avg_loss_this_rank = total_loss_epoch_accum / len(dataloader) if len(dataloader) > 0 else 0.0 # Average over number of micro-batches processed
        
        if world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss_this_rank, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss_epoch_global = avg_loss_tensor.item()
        else:
            avg_loss_epoch_global = avg_loss_this_rank
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if rank == 0:
            logging.info(
                f"Epoch {epoch+1} Summary: Avg Global Loss (per micro-batch): {avg_loss_epoch_global:.4f} | "
                f"LR: {current_lr:.2e} | OptSteps: {num_optimizer_steps_epoch}"
            )
            # MODIFICATION: Log empty batch ratio for the epoch
            empty_batch_ratio = total_empty_batches_in_epoch / len(dataloader) if len(dataloader) > 0 else 0
            logging.info(f"Epoch {epoch+1}: Empty batch ratio: {empty_batch_ratio:.2%} ({total_empty_batches_in_epoch}/{len(dataloader)})")
            if empty_batch_ratio > EMPTY_BATCH_EPOCH_THRESHOLD:
                logging.critical(
                    f"Epoch {epoch+1}: High empty batch ratio ({empty_batch_ratio:.2%}) exceeded threshold ({EMPTY_BATCH_EPOCH_THRESHOLD:.0%}). "
                    "Strongly consider checking data quality."
                )
            
            if avg_loss_epoch_global < best_loss:
                best_loss = avg_loss_epoch_global
                logging.info(f"Best global loss updated: {best_loss:.4f}. Saving model to {model_save_path_best}")
                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(state_to_save, model_save_path_best)
            
            checkpoint_path = current_save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss_epoch_global,
                'best_loss': best_loss,
                'huber_delta': huber_delta,
                'args': args
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    if world_size > 1:
        cleanup_ddp()
    if rank == 0:
        logging.info(f"Training complete. Best model saved at {model_save_path_best}")
        logging.info(f"Final logs and checkpoints are in {current_log_dir.parent}")

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
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # This typically means it's already been set, or 'spawn' is not supported/default on the system.
        # Forcing can be problematic on some systems if not the first call.
        # Default 'fork' on Linux is usually fine for DDP with CUDA if globals are handled carefully.
        # 'spawn' is generally safer cross-platform and for CUDA.
        print("Note: Multiprocessing start method 'spawn' was already set or could not be forced. Continuing.")
        pass 
    # main()