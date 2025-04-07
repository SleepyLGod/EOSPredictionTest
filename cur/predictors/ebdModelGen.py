import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

import math
import json
import bisect
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from datetime import datetime
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, utils

torch.backends.cuda.max_split_size_mb = 128 # set max split size to avoid fragmentation
torch.cuda.set_per_process_memory_fraction(0.9) # limit GPU memory usage

HIDDEN_SIZE = 8192  # Llama-3 hidden size: 8192 = 1024 * 8
DROPOUT_RATE = 0.3
BATCH_SIZE = 128    # batch_size
LEARNING_RATE = 1e-3
EPOCHS = 50         # more epoch to learn embedding features
REG_WEIGHT = 0.6    # [0.0-1.0]
CLS_WEIGHT = 0.4    # automatically calculated to 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CUR_DIR = Path(__file__).parent.absolute()
FEATURE_DIR = CUR_DIR.parent / "training_data" / "ebd" / "features" / "llama3_70b"
METADATA_DIR = CUR_DIR.parent / "training_data" / "ebd" / "metadata" / "llama3_70b"

if not FEATURE_DIR.exists():
    raise FileNotFoundError(f"feature directory non-exist: {FEATURE_DIR}")

# logging configuration
LOG_DIR = CUR_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()] # log to both file and console
)

SAVE_DIR = CUR_DIR / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)
model_save_path = SAVE_DIR / "enhanced_mlp.pth"

def decode_params(encoded):
    encoded = int(encoded)
    return {
        'temperature': ((encoded >> 24) & 0xFF) / 255 * 0.8 + 0.1,
        'top_k': (encoded >> 17) & 0x7F,
        'repetition_penalty': ((encoded >> 9) & 0xFF) / 255 * 0.3 + 1.3,
        'max_new_tokens': ((encoded >> 6) & 0x7) * 100
    }

class EmbeddingDataset(Dataset):
    def __init__(self, feature_dir):
        self.file_paths = []
        self.cumulative_samples = []
        self.file_sample_counts = []
        
        # 严格校验每个文件
        sorted_files = sorted(os.listdir(feature_dir))
        for f in sorted_files:
            file_path = os.path.join(feature_dir, f)
            if not file_path.endswith('.npz'):
                continue
                
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    # 必须包含features和labels
                    assert 'features' in data and 'labels' in data, "Missing required arrays"
                    features = data['features']
                    labels = data['labels']
                    # 非空校验
                    assert len(features) > 0, "Empty features"
                    assert len(labels) > 0, "Empty labels"
                    # 长度一致性校验
                    assert len(features) == len(labels), "Length mismatch"
                    assert features.dtype == object and labels.dtype == object, "Expected object dtype"
                    if len(features) > 0:
                        assert isinstance(features[0], dict), "Features should be dict"
                        assert isinstance(labels[0], dict), "Labels should be dict"
                    n_samples = len(features)
                    self.file_paths.append(file_path)
                    self.file_sample_counts.append(n_samples)
            except Exception as e:
                logging.error(f"INVALID FILE {file_path}: {str(e)}")
                continue  # 跳过无效文件
        
        # empty file check
        if not self.file_paths:
            raise RuntimeError(f"数据集目录 {feature_dir} 中没有有效文件")
        # calculate cumulative samples
        total = 0
        for i, path in enumerate(self.file_paths):
            n = self.file_sample_counts[i]
            self.cumulative_samples.append((total, total + n))
            # logging.debug(f"File {path}: samples={n}, range=({total}, {total + n})")
            total += n
        self.total_samples = total
        logging.info(f"Total files: {len(self.file_paths)}, Total samples: {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx >= self.total_samples or idx < 0:
            logging.error(f"Invalid idx {idx}, total_samples={self.total_samples}")
            return None
        # get the file index and sample index
        start_list = [cs[0] for cs in self.cumulative_samples]
        file_idx = bisect.bisect_right(start_list, idx) - 1
        if file_idx < 0 or file_idx >= len(self.file_paths):
            logging.error(f"Invalid file_idx {file_idx} for idx {idx}")
            return None
        start, end = self.cumulative_samples[file_idx]
        sample_idx = idx - start
        if sample_idx >= self.file_sample_counts[file_idx]:
            logging.error(f"IndexError: sample_idx {sample_idx} >= {self.file_sample_counts[file_idx]} in file {self.file_paths[file_idx]}")
            return None
        try:
            with np.load(
                self.file_paths[file_idx], 
                allow_pickle=True, 
                mmap_mode='r'
            ) as data:
                feat_dict = data['features'][sample_idx]
                lab_dict = data['labels'][sample_idx]
                
                required_feat_keys = ['sys_para', 'pos', 'ebd']
                required_lab_keys = ['rest_len', 'over_max_len']
                for key in required_feat_keys:
                    if key not in feat_dict:
                        raise KeyError(f"Missing key {key} in features")
                for key in required_lab_keys:
                    if key not in lab_dict:
                        raise KeyError(f"Missing key {key} in labels")
                
                encoded_param = feat_dict['sys_para']
                seq_pos = feat_dict['pos']
                embedding = feat_dict['ebd']
                rest_len = lab_dict['rest_len']
                over_max = lab_dict['over_max_len']
                
                params = decode_params(encoded_param)
                
                return {
                    'temperature': np.float32((params['temperature'] - 0.1) / 0.8),
                    'top_k': np.float32(params['top_k'] / 100.0),
                    'repetition_penalty': np.float32((params['repetition_penalty'] - 1.3) / 0.3),
                    'max_len': np.float32((params['max_new_tokens'] - 300) / 200.0),  
                    'seq_pos': np.float32(seq_pos / 4096.0),
                    'embedding': embedding.astype(np.float32),
                    'remaining': np.float32(rest_len),
                    'over_max': np.float32(over_max)
                }
        except Exception as e:
                error_info = f"""
                Error Details:
                - File: {self.file_paths[file_idx]}
                - Sample Index: {sample_idx}
                - Original Error: {str(e)}
                """
                logging.error(error_info)
                return None

class EnhancedMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # feature interaction
        self.interaction = nn.Sequential(
            nn.Linear(hidden_size + 5, 2048),  # 5 parameters
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        # multi-task heads
        self.reg_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        
        # residual connection
        self.residual = nn.Linear(hidden_size + 5, 1024)
    
    def forward(self, x):
        # check feature dimensions
        for param in ['temperature', 'top_k', 'repetition_penalty', 'max_len', 'seq_pos']:
            assert x[param].dim() == 1, f"{param} should be 1D tensor"
        
        # concatenate features
        main_feature = torch.cat([
            x['embedding'],
            x['temperature'].unsqueeze(1),
            x['top_k'].unsqueeze(1),
            x['repetition_penalty'].unsqueeze(1),
            x['max_len'].unsqueeze(1),
            x['seq_pos'].unsqueeze(1)
        ], dim=1)
        
        # feature interaction
        interacted = self.interaction(main_feature)
        
        # residual connection
        residual = self.residual(main_feature)
        fused = interacted + residual
        
        # multi-task heads
        reg_out = self.reg_head(fused).squeeze(-1)
        cls_out = self.cls_head(fused).squeeze(-1)
        return reg_out, cls_out

def train_step(model, batch, reg_criterion, cls_criterion, reg_weight=REG_WEIGHT, cls_weight=CLS_WEIGHT):
    # move data to device
    device_batch = {k: v.to(DEVICE) for k, v in batch.items()}
    reg_labels = device_batch.pop('remaining')
    cls_labels = device_batch.pop('over_max')
    
    # forward pass
    reg_pred, cls_pred = model(device_batch)
    
    # loss calculation
    reg_loss = reg_criterion(reg_pred, reg_labels)
    cls_loss = cls_criterion(cls_pred, cls_labels)
    total_weight = reg_weight + cls_weight
    return (reg_weight/total_weight)*reg_loss + (cls_weight/total_weight)*cls_loss

def collate_fn(batch):
    original_size = len(batch)
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    if len(batch) == 0:
        raise RuntimeError("Empty batch after filtering")
    filtered = original_size - len(batch)
    if filtered > 0:
        logging.warning(f"Filtered {filtered}/{original_size} samples from batch")
        if filtered / original_size > 0.1:  # Example threshold: 10%
            raise RuntimeError(f"Too many failed samples ({filtered}/{original_size})")
    collated = torch.utils.data.dataloader.default_collate(batch)
    if not torch.isfinite(collated['embedding']).all():
        logging.error("NaN or Inf detected in collated batch")
        return {}
    for key in ['temperature', 'top_k', 'repetition_penalty', 'max_len', 'seq_pos']:
        if collated[key].min() < 0.0 or collated[key].max() > 1.0:
            logging.error(f"Invalid value in {key}: min={collated[key].min()}, max={collated[key].max()}")
            return {}
    return collated

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
    
    # data loading
    dataset = EmbeddingDataset(FEATURE_DIR)
    
    pos_count = 0
    rest_lengths = []
    error_files = []
    for path in tqdm(dataset.file_paths, desc="加载数据"):
        try:
            with np.load(path, allow_pickle=True) as data:
                labs = data['labels']
                # 遍历 labs 中的每个字典
                for lab in labs:
                    rest_len = lab['rest_len']  # 直接访问字典键
                    over_max = lab['over_max_len']  # 使用测试中确认的键名
                    pos_count += int(over_max)  # 转换为整数（True -> 1, False -> 0）
                    rest_lengths.append(rest_len)
        except Exception as e:
            error_files.append(path)
            logging.error(f"处理文件失败 {path}: {str(e)}")
            continue  # 继续处理下一个文件

    # 空数据详细报告
    if not rest_lengths:
        logging.critical("以下文件导致数据加载失败:")
        for f in error_files:
            logging.critical(f" - {f}")
        raise RuntimeError("所有数据文件均无效，请检查上述文件")
    
    dataloader = DataLoader(
        dataset, 
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        drop_last=False # whether to drop the last incomplete batch
    )
    
    # model initialization
    model = EnhancedMLP(HIDDEN_SIZE)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)  # Move after DataParallel if needed
    
    # optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-4, 
        betas=(0.9, 0.999)
    )
    
    # total_steps = EPOCHS * ((len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps = EPOCHS * len(dataloader)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2e-3, 
        total_steps=total_steps, # total_steps=EPOCHS*len(dataloader),
        pct_start=0.3,
        div_factor=10,  # 初始学习率降为max_lr/10
        final_div_factor=100  # 最终学习率降为max_lr/100
    )
    logging.info(f"Total training steps: {total_steps}")
    
    # setting for loss functions
    pos_weight = (len(rest_lengths) - pos_count) / max(pos_count, 1e-6)
    try:
        rest_arr = np.array(rest_lengths, dtype=np.float32)
        if len(rest_arr) == 0:
            raise ValueError("空数组")
            
        rest_mean = np.nanmean(rest_arr)  # 忽略NaN
        abs_diff = np.abs(rest_arr - rest_mean)
        delta = np.nanquantile(abs_diff, 0.9)  # 忽略NaN
    except Exception as e:
        delta = 1.0
        logging.warning(f"使用安全delta值1.0,原因: {str(e)}")
    
    # loss functions
    reg_criterion = nn.HuberLoss(delta=float(delta))
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    
    # training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            loss = train_step(
                model, 
                batch, 
                reg_criterion, 
                cls_criterion, 
                reg_weight=REG_WEIGHT, 
                cls_weight=CLS_WEIGHT
            )
            loss.backward()
            
            # grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            # if grad_norms:
            #     max_norm = np.percentile(grad_norms, 90)
            #     max_norm = max(0.5, min(max_norm, 5.0))  # clamp to [0.5, 5.0]
            # else:
            #     max_norm = 1.0
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            # gradient monitoring
            if batch_idx % 50 == 0 and logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
                logging.debug(f"Grad Norm: {total_norm.item():.2f}")
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            logging.info(f"Best loss updated: {best_loss:.4f}")
            torch.save(
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                model_save_path
            )
        logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        checkpoint_path = SAVE_DIR / f"checkpoint_epoch_{epoch+1}.pth"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")

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
    problem_file = "meta-llama_Meta-Llama-3-70B_t0.1_tk100_r1.6_mok500_qa50087.npz"  # 替换为实际文件名
    path = os.path.join(FEATURE_DIR, problem_file)
    with np.load(path, allow_pickle=True) as data:
        features = data['features']
        labels = data['labels']
        print(f"File: {path}")
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        # 检查所有样本的键
        for i in range(len(features)):
            try:
                feat = features[i]
                lab = labels[i]
                assert all(k in feat for k in ['sys_para', 'pos', 'ebd'])
                assert all(k in lab for k in ['rest_len', 'over_max_len'])
            except Exception as e:
                print(f"Invalid sample {i}: {str(e)}")

if __name__ == "__main__":
    # datacheck()
    # inspect_problem_file()
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    torch.cuda.empty_cache()
    # check_dataset(FEATURE_DIR)
    logging.info(f"Available GPUs: {torch.cuda.device_count()}")
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    # main()

    # # save the model
    # if isinstance(model, nn.DataParallel):
    #     torch.save(model.module.state_dict(), model_save_path) # multi-GPU
    # else:
    #     torch.save(model.state_dict(), model_save_path)  # single GPU
    # logging.info(f"model saved to: {model_save_path}")

# class EmbeddingDataset(Dataset):
#     def __init__(self, feature_dir):
#         self.samples = []
#         # index loading (in parellel)
#         from concurrent.futures import ThreadPoolExecutor
#         with ThreadPoolExecutor(max_workers=8) as executor:
#             futures = []
#             for fname in os.listdir(feature_dir):
#                 if not fname.endswith('.npz'):
#                     continue
#                 path = os.path.join(feature_dir, fname)
#                 futures.append(executor.submit(self._load_file, path))
            
#             for future in tqdm(futures, desc="Loading files"):
#                 self.samples.extend(future.result())
    
#     def _load_file(self, path):
#         samples = []
#         with np.load(path, allow_pickle=True) as data:
#             features = data['features']
#             labels = data['labels']
#             for feat, lab in zip(features, labels):
#                 # decoding parameters
#                 params = decode_params(feat['system_params'].item())
                
#                 # normalizing features
#                 norm_features = {
#                     'temperature': (params['temperature'] - 0.1) / 0.8,
#                     'top_k': params['top_k'] / 100.0,
#                     'repetition_penalty': (params['repetition_penalty'] - 1.3) / 0.3,
#                     'max_len': (params['max_new_tokens'] - 300) / 200.0,
#                     'seq_pos': feat['seq_pos'].item() / 4096.0,
#                     'embedding': feat['embedding'].astype(np.float32),  # float16->float32
#                     'remaining': lab['rest_len'],
#                     'over_max': float(lab['over_max_len'])
#                 }
#                 samples.append(norm_features)
#         return samples
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         return {
#             'temperature': torch.tensor(sample['temperature'], dtype=torch.float32),
#             'top_k': torch.tensor(sample['top_k'], dtype=torch.float32),
#             'repetition_penalty': torch.tensor(sample['repetition_penalty'], dtype=torch.float32),
#             'max_len': torch.tensor(sample['max_len'], dtype=torch.float32),
#             'seq_pos': torch.tensor(sample['seq_pos'], dtype=torch.float32),
#             'embedding': torch.tensor(sample['embedding'], dtype=torch.float32),
#             'remaining': torch.tensor(sample['remaining'], dtype=torch.float32),
#             'over_max': torch.tensor(sample['over_max'], dtype=torch.float32)
#         }