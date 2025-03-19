'''
# Description: This file contains the data structure for the dataset generator.
{
    "system_params": uint32,  # encoded
    "seq_pos": uint16,        
    "logits": float16[VocabSize],  # half precision
    "label": {
        "remaining_tokens": int,     # remaining tokens
        "over_max_seq_len": bool     # over max sequence length?
    }
}
'''

import json
import torch
import re
import os
import math
import random
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, utils
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
utils.logging.set_verbosity_error()

# Model and dataset config
models = [
        'meta-llama/Llama-3.2-1B', # Llama-3.2-1B (16 layers 32 heads)
        'meta-llama/Meta-Llama-3-8B', # Meta-Llama-3-8B
        'meta-llama/Meta-Llama-3-70B', # Meta-Llama-3-70B
        'meta-llama/Meta-Llama-3-8B-Instruct', # Meta-Llama-3-8B-Instruct (the only model TRAIL tests with)
        'EleutherAI/gpt-j-6b', # GPT-J 
        'meta-llama/Llama-2-13b-chat-hf', # Llama-2, fine-tuned
        'EleutherAI/gpt-neox-20b', # GPT-NeoX
        "CreitinGameplays/gpt3-finnish-xl-alpaca", # GPT3-Finnish-XL-ALPACA
        "chavinlo/gpt4-x-alpaca", # GPT4-X-ALPACA
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # DeepSeek-R1-Distill-Qwen-1.5B
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", # DeepSeek-R1-Distill-Qwen-14B
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # DeepSeek-R1-Distill-Qwen-32B
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # DeepSeek-R1-Distill-Llama-8B
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", # DeepSeek-R1-Distill-Llama-70B
        ]

CACHE_DIR = "./.cache/huggingface/datasets"
FEATURE_DIR = "./training_data/logits_as/features/llama3_70b"
METADATA_DIR = "./training_data/logits_as/metadata/llama3_70b"
DS_NAME = 'yahma/alpaca-cleaned'
TOP_K = 1000
BATCH_SIZE = 4 

# Generation parameters
temperatures = [0.1, 0.3, 0.5, 0.9]
top_k_values = [1, 5, 10, 100]
repetition_penalties = [1.3, 1.6]
max_new_tokens_values = [300, 500]

# Load dataset
ds = load_dataset(DS_NAME)
dataset = ds['train']
prompts = [f"{inst}\n\n{inp}" if inp.strip() else inst for inst, inp in zip(dataset["instruction"], dataset["input"])]

data = {
    "qa_pairs": [
        {"prompt": p, "response": o} for p, o in zip(prompts, dataset["output"])
    ]
}

# Load model
model_choice = 3
model_name = models[model_choice - 1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    import flash_attn
    has_flash_attn = True
    attn_implementation = "flash_attention_2"
    print("Flash Attention 2 is available")
except ImportError:
    has_flash_attn = False
    attn_implementation = "eager" 
    print("Flash Attention 2 is not installed - falling back to default attention")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,  # Enable low CPU memory usage
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    output_hidden_states=True # Required for feature extraction
)

# Utility functions
def sample_top_k(logits, k):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    return top_k_indices[0, torch.multinomial(probs, 1).item()]

def encode_params(params):
    # T（0.1~0.9 -> 0~255）
    temp_enc = int(np.interp(params['temperature'], [0.1, 0.9], [0, 255]))
    # top_k
    assert 1 <= params['top_k'] <= 127, f"top_k {params['top_k']} out of range"
    topk_enc = params['top_k']
    # rp（1.3~1.6 -> 0~255）
    rep_enc = int(np.interp(params['repetition_penalty'], [1.3, 1.6], [0, 255]))
    # max_new_tokens
    maxlen_enc = params['max_new_tokens'] // 100
    assert 3 <= maxlen_enc <= 5, f"max_new_tokens {params['max_new_tokens']} invalid"
    
    # distribution: T(8) top_k(7) rp(8) maxlen(3)
    return np.uint32(
        (temp_enc << 24) | 
        (topk_enc << 17) |  # bit17~bit23（7'）
        (rep_enc << 9) |    # bit9~bit16（8'）
        (maxlen_enc << 6)   # bit6~bit8（3'）
    )

def decode_params(encoded):
    encoded = int(encoded)
    return {
        'temperature': ((encoded >> 24) & 0xFF) / 255 * 0.8 + 0.1,
        'top_k': (encoded >> 17) & 0x7F,  # 0x7F=01111111
        'repetition_penalty': ((encoded >> 9) & 0xFF) / 255 * 0.3 + 1.3,
        'max_new_tokens': ((encoded >> 6) & 0x7) * 100  # 0x7=0111
    }

def process_batch(inputs, param):
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=param['max_new_tokens'],
        temperature=param['temperature'],
        top_k=param['top_k'],
        repetition_penalty=param['repetition_penalty'],
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return outputs

def prompt_selection():
    return random.random() < 0.01  # 1% of prompts

# Dataset generator
def ds_generator(model, tokenizer, data, device):
    model_name = model.config.name_or_path.replace('/', '_')
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    param_combinations = [
        {'temperature': t, 'top_k': k, 'repetition_penalty': rp, 'max_new_tokens': msl} 
        for t in temperatures for k in top_k_values
        for rp in repetition_penalties for msl in max_new_tokens_values
    ]

    pbar = tqdm(total=len(param_combinations)*len(data['qa_pairs']), desc="Generating Dataset")

    for param in param_combinations:
        for qa_idx, qa in enumerate(data['qa_pairs']):
            if not prompt_selection():
                continue
            try:
                prompt = qa['prompt']
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
                prompt_len = input_ids.shape[1]
                generated_tokens = []
                generated_tokens_set = set()
                features = []
                eos_encountered = False
                
                # Generation loop
                past_key_values = None
                for step in range(param['max_new_tokens']):
                    outputs = model(
                        input_ids if step == 0 else input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    temp_scaled_logits = outputs.logits[:, -1, :]
                    if param['repetition_penalty'] != 1.0:
                        for token in generated_tokens_set:
                            temp_scaled_logits[0, token] /= param['repetition_penalty']
                    temp_scaled_logits /= param['temperature'] # temperature scaling
                    # logits_np = logits.detach().cpu().numpy().astype(np.float16)
                    # data compression
                    topk_values, topk_indices = torch.topk(temp_scaled_logits, TOP_K, dim=-1)
                    
                    seq_pos = input_ids.shape[1] - 1
                    cur_step = step + 1
                    
                    vocab_size = tokenizer.vocab_size
                    dtype = np.uint32 if vocab_size > 65535 else np.uint16
                    # Store features
                    features.append({
                        'system_params': encode_params(param),
                        'seq_pos': np.uint32(seq_pos),
                        "step": np.uint32(cur_step),
                        'topk_values': topk_values[0].detach().cpu().numpy().astype(dtype),
                        'topk_indices': topk_indices[0].detach().cpu().numpy().astype(dtype)
                    })
                    
                    # final top-k sampling
                    next_token = sample_top_k(temp_scaled_logits, param['top_k'])
                    generated_tokens.append(next_token.item())
                    generated_tokens_set.add(next_token.item())

                    # Check EOS
                    if next_token == tokenizer.eos_token_id:
                        eos_encountered = True
                        break

                    # Update input
                    input_ids = torch.cat([input_ids, next_token.view(1,1)], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones(1,1, device=device)], dim=1)

                # Generate labels
                labels = []
                total_steps = step + 1 if eos_encountered else param['max_new_tokens']
                for feat in features:
                    if eos_encountered:
                        remaining = (prompt_len + total_steps) - int(feat['seq_pos']) - 1
                        over_max = False
                    else:
                        remaining = param['max_new_tokens'] - (int(feat['seq_pos']) - prompt_len)
                        over_max = remaining < 0
                    labels.append({
                        'remaining_tokens': max(0, remaining),
                        'over_max_seq_len': over_max
                    })

                # Save features and labels
                if features:
                    filename = (
                        f"{model_name}_t{param['temperature']}_tk{param['top_k']}"
                        f"_r{param['repetition_penalty']}_mok{param['max_new_tokens']}_qa{qa_idx}.npz"
                    )
                    np.savez_compressed(
                        os.path.join(FEATURE_DIR, filename),
                        features=np.array(features, dtype=object),
                        labels=np.array(labels, dtype=object)
                    )

                    # Save metadata
                    metadata = {
                        'prompt': prompt,
                        'generated_tokens': generated_tokens,
                        'generated_text': tokenizer.decode(generated_tokens).replace('\n', ' '),
                        'total_steps': total_steps,
                        'eos_encountered': eos_encountered,
                        'sample_feature': {
                            'system_params': int(features[0]['system_params']),
                            'seq_pos': int(features[0]['seq_pos']),
                            'topk_values': features[0]['topk_values'].tolist(),
                            'topk_indices': features[0]['topk_indices'].tolist()
                        },
                        'sample_label': {
                            'remaining_tokens': int(labels[0]['remaining_tokens']),
                            'over_max_seq_len': bool(labels[0]['over_max_seq_len'])
                        }
                    }
                    with open(os.path.join(METADATA_DIR, f"{filename[:-4]}.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)

                del outputs
                torch.cuda.empty_cache()
                if past_key_values:
                    del past_key_values
                pbar.update(1)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM at {qa_idx} with param {param}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    pbar.close()

# Execute
ds_generator(model, tokenizer, data, device)
print("Dataset generation completed.")