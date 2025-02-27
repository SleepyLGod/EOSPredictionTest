import json
import torch
import re
import os
import math
import random
from tqdm import tqdm
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, utils
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

utils.logging.set_verbosity_error()  # Suppress standard warnings

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

datasets = {
        1: './data/dataset_alpaca.json',
        2: './data/datasetSimplified_alpaca.json',
        3: './data/dataset_lmsys-chat-1m.json',
        4: 'yahma/alpaca-cleaned'
}

SPECIAL_LABEL = -1
CACHE_DIR = "./.cache/huggingface/datasets"
FEATURE_DIR = "./training_data/features/llma3_1b"
METADATA_DIR = "./training_data/metadata/llma3_1b"
DS_NAME = datasets[4]

temperatures = [0.1, 0.3, 0.5, 0.9] # low, mid, high creativity
top_k_values = [1, 5, 10] # low, mid, high diversity
repetition_penalties = [1.3, 1.6] # low, mid, high coherence
max_new_tokens_values = [300, 500] # low, mid, high length
# system_parameters = []
# for temp in temperatures:
#         for k in top_k_values:
#                 for rep_pen in repetition_penalties:
#                         for max_tok in max_new_tokens_values:
#                                 system_parameters.append({
#                                         'temperature': temp,
#                                         'top_k': k,
#                                         'repetition_penalty': rep_pen,
#                                         'max_new_tokens': max_tok
#                                 })
# print(f"Total number of system parameters: {len(system_parameters)}")

TEMP_SAMP = temperatures[1]
TK_SAMP = top_k_values[2]
REP_PEN_SAMP = repetition_penalties[2]
MAX_TOKEN_LEN_SAMP = max_new_tokens_values[1]

# Parameters
max_new_tokens = MAX_TOKEN_LEN_SAMP # Maximum number of tokens to generate
temperature = TEMP_SAMP  # Lower temperature for more deterministic output
top_k = TK_SAMP  # Increase top_k for more diverse candidates
repetition_penalty = REP_PEN_SAMP  # Increase repetition penalty to reduce repetition

ds = load_dataset(DS_NAME)
dataset = ds['train']

prompt_template = "{instruction}\n\n{input}"  # template for the prompt
prompts = []

for inst, inp in zip(dataset["instruction"], dataset["input"]):
    if inp.strip() == "":  # no input
        prompt = inst
    else:
        prompt = prompt_template.format(instruction=inst, input=inp)
    prompts.append(prompt)

# structure the data
data = {
    "qa_pairs": [
        {"prompt": prompt, "response": output}
        for prompt, output in zip(prompts, dataset["output"])
    ]
}

# print the output
# print(json.dumps(data, indent=2))

# Select model
# print("Choose the model to test:")
# for i, model in enumerate(models, 1):
#     print(f" {i}. {model.split('/')[-1]}")
model_choice = 1
# if model_choice < 1 or model_choice > len(models):
#     raise ValueError(f"Invalid model choice. Please enter a number between 1 and {len(models)}")
model_name = models[model_choice - 1]

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=CACHE_DIR,
    device_map="auto",
    attn_implementation="eager"
)

max_seq_len = tokenizer.model_max_length
eos_token_id = tokenizer.eos_token_id
eos_flag=0

def sample_top_k(logits, k, temperature):
    logits = logits / temperature
    # top_k = min(k, logits.size(-1))
    top_k_logits, top_k_indices = torch.topk(logits, k) # Get top k candidates
    probs = F.softmax(top_k_logits, dim=-1)
    return top_k_indices[0, torch.multinomial(probs, 1).item()]

def commpress_attention_matrix(attention_matrix):
    """
    half precision
    sparse matrix
    Not yet: Compress the attention matrix by removing the diagonal and upper triangular part
    Not yet: SVG compression
    """
    seq_len = attention_matrix.shape[0]
    mask = np.tri(seq_len, dtype=bool)
    attention_matrix = attention_matrix * mask
    
    attention_matrix = attention_matrix.astype(np.float16)
    sparse_matrix = sparse.coo_matrix(attention_matrix)
    return sparse_matrix
    
    # seq_len = attention_matrix.shape[0]
    # mask = np.triu(np.ones((seq_len, seq_len)), k=0)
    # attention_matrix = attention_matrix * (1 - mask)
    
    # return {
    #     'data': sparse_matrix.data,
    #     'row': sparse_matrix.row,
    #     'col': sparse_matrix.col,
    #     'shape': sparse_matrix.shape
    # }

def encode_params(params):
    """
    Encode system parameters into a numpy array / 32-bit integer
    """
    # return np.array([
    #     np.interp(params['temperature'], [0.1, 0.9], [0, 255]),
    #     params['top_k'],
    #     np.interp(params['repetition_penalty'], [1.0, 1.6], [0, 255]),
    #     params['max_seq_len'] // 100
    # ], dtype=np.uint8)
    assert 0.1 <= params['temperature'] <= 0.9
    assert 1 <= params['top_k'] <= 255
    assert 1.0 <= params['repetition_penalty'] <= 1.6
    assert 100 <= params['max_seq_len'] <= 25500
    
    # encode
    temp_enc = int(np.interp(params['temperature'], [0.1, 0.9], [0, 255]))
    topk_enc = params['top_k']
    rep_enc = int(np.interp(params['repetition_penalty'], [1.0, 1.6], [0, 255]))
    maxlen_enc = params['max_seq_len'] // 100
    
    return np.uint32(
        (temp_enc << 24) | 
        (topk_enc << 16) | 
        (rep_enc << 8) | 
        maxlen_enc
    )

def decode_params(encoded):
    """
    Decode system parameters from a numpy array / 32-bit integer
    """
    return {
        'temperature': ((encoded >> 24) & 0xFF) / 255 * 0.8 + 0.1,
        'top_k': (encoded >> 16) & 0xFF,
        'repetition_penalty': ((encoded >> 8) & 0xFF) / 255 * 0.6 + 1.0,
        'max_seq_len': (encoded & 0xFF) * 100
    }

def head_selection():
    """
    Select a head from the model, randomly chose about 25% of the heads
    """
    return random.random() < 0.25

def layer_selection(layer_idx):
    """s
    for deeper layers, have a higher chance to be selected
    """
    total_layers = model.config.num_hidden_layers
    if layer_idx >= total_layers * 0.5:
        return True
    return random.random() < 0.5
    
    
# ======================
# Generate training data
# ======================
"""
dataset structure:
{
    "system_params": {
        "temperature": float,
        "top_k": int,
        "repetition_penalty": float,
        "max_seq_len": int
    },
    "model_arch": {
        "num_layers": int,
        "num_heads": int
    },
    "samples": [
        {
            "layer": int,
            "head": int,
            "attention_matrix": np.array(seq_len, seq_len),
            "seq_pos": int,
        },
        ... # other head samples
    ],
    "label": {
        "remaining_tokens": int,
        "over_max_seq_len": bool
    }
}
"""

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def ds_generator(model, tokenizer, data, device):
    model_name = model.config.name_or_path.replace('/', '_')
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Model: {model_name}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of heads: {num_heads}")
    print(f"eos_token_id: {eos_token_id}")
    
    # system_params 
    param_combinations = [
        {
            "temperature": t, 
            "top_k": k, 
            "repetition_penalty": rep_pen, 
            "max_seq_len": msl
            }
        for t in temperatures
        for k in top_k_values
        for rep_pen in repetition_penalties
        for msl in max_new_tokens_values
    ]
    print(f"Total number of system parameters: {len(param_combinations)}")
    
    # loading bar initialization
    pbar = tqdm(
                total=len(param_combinations)*len(data['qa_pairs']), 
                desc="Generating Dataset"
            )

    # traverse through all system parameters
    for param in param_combinations:
        # traverse dataset
        for qa_idx, qa in enumerate(data['qa_pairs']):
            try:
                # initialize model
                prompt = qa['prompt']
                target_response = qa.get('response', '')
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                prompt_len = input_ids.shape[1]
                generated_tokens = []
                features = []
                seq_poses = []
                labels = []
                
                # token generation loop
                for step in range(param['max_seq_len']):
                    with torch.no_grad():
                        outputs = model(
                            input_ids,
                            attention_mask=attention_mask,
                            output_attentions=True
                        )
                    
                    # feature extraction
                    current_seq_len = input_ids.shape[1] # including prompt size
                    seq_pos = current_seq_len - 1  # current sequence position
                    seq_poses.append(seq_pos)
                    
                    # attention matrix extraction
                    for layer_idx in range(num_layers):
                        if not layer_selection(layer_idx):
                            continue
                        # shape: (batch_size=1, num_heads, seq_len, seq_len)
                        layer_attn = outputs.attentions[layer_idx]
                        
                        for head_idx in range(num_heads):
                            if not head_selection():
                                continue
                            # shape: (seq_len, seq_len)
                            attn_matrix = layer_attn[0, head_idx].cpu().numpy()
                            
                            # sample structure
                            feature = {
                                "system_params": param,
                                "model_arch": {
                                    "num_layers": num_layers,
                                    "num_heads": num_heads
                                },
                                "layer": layer_idx,
                                "head": head_idx,
                                "attention_matrix": attn_matrix,
                                "seq_pos": seq_pos,
                                # "remaining_tokens": None  
                            }
                            features.append(feature)
                            print(f"Generated feature {len(features)}")

                    # ===== token generation =====
                    logits = outputs.logits[:, -1, :]
                    
                    # repetition penalty application
                    if param['repetition_penalty'] != 1.0:
                        unique_tokens = set(generated_tokens)
                        for token in unique_tokens:
                            logits[0, token] /= param['repetition_penalty']

                    # sample next token
                    next_token = sample_top_k(
                        logits, 
                        param['top_k'],
                        param['temperature']
                    )

                    # EOS token encountered
                    if next_token.item() == eos_token_id: # EOS token encountered
                        # calculate remaining tokens of each feature
                        total_generated = len(generated_tokens) + 1 + prompt_len # including prompt size and current token
                        for feature in features:
                            label = {}
                            label['remaining_tokens'] = max(0, total_generated - feature['seq_pos'] - 1)
                            label['over_max_seq_len'] = False
                            labels.append(label)
                        break
                        
                    # input updating
                    input_ids = torch.cat(
                        [input_ids, next_token.view(1, 1)], 
                        dim=1
                    )
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones(1, 1, device=device)], 
                        dim=1
                    )
                    generated_tokens.append(next_token.item())
                    
                else: # if the loop is not broken, then execute this block
                    # cannot meet EOS token
                    for feature in features:
                        label = {}
                        label['remaining_tokens'] = max(0, param['max_seq_len'] - feature['seq_pos'])
                        label['over_max_seq_len'] = True
                        labels.append(label)

                # ========== data saving ==========
                if features:
                    # # structure data
                    # feature_data = {
                    #     "system_params": param,
                    #     "model_arch": {
                    #         "num_layers": num_layers,
                    #         "num_heads": num_heads
                    #     },
                    #     "samples": features,
                    #     "labels": labels
                    # }
                    
                    # file name generation
                    model_name = model.config.name_or_path.replace('/', '_')
                    filename = f"{model_name}_temp{param['temperature']}_topk{param['top_k']}_rep{param['repetition_penalty']}_seq{qa_idx}.npz"
                    
                    # # feature data saving
                    # # np.savez_compressed(
                    # #     os.path.join(FEATURE_DIR, filename),
                    # #     data=feature_data
                    # # )
                    # np.savez_compressed(
                    #     os.path.join(FEATURE_DIR, filename),
                    #     system_params=param,
                    #     model_arch={
                    #         "num_layers": num_layers,
                    #         "num_heads": num_heads
                    #     },
                    #     features=features,
                    #     labels=labels
                    # )
                    compressed_features = []
                    for feature in features:
                        compressed_attn = commpress_attention_matrix(feature["attention_matrix"])
                        encoded_params = encode_params(feature["system_params"])
                        
                        compressed_feature = {
                            "system_params": encoded_params,
                            "model_arch": feature["model_arch"],
                            "layer": np.uint8(feature["layer"]), # 0-15
                            "head": np.uint8(feature["head"]), # 0-31
                            # 'attn_data': compressed_attn['data'],
                            # 'attn_row': compressed_attn['row'].astype(np.uint16),
                            # 'attn_col': compressed_attn['col'].astype(np.uint16),
                            # 'attn_shape': (np.uint16(compressed_attn['seq_len']),) * 2,
                            "attn_data": compressed_attn.data,
                            "seq_pos": np.uint16(feature["seq_pos"]) # 0-25500
                        }
                        compressed_features.append(compressed_feature)
                    
                    np.savez_compressed(
                        os.path.join(FEATURE_DIR, filename),
                        features=np.array(compressed_features, dtype=object),
                        labels=labels,
                        param_interp_ranges={
                            'temperature': [0.1, 0.9],
                            'rep_penalty': [1.0, 1.6]
                        }
                    )

                    # metadata saving
                    metadata = {
                        "prompt": prompt,
                        "response": target_response,
                        "model_response": tokenizer.decode(generated_tokens, skip_special_tokens=True),
                        "generated_tokens": generated_tokens,
                        "prompt_len": prompt_len,
                        "total_steps": step + 1,
                        "eos_encountered": next_token.item() == eos_token_id,
                        # randomly choose one feature and its related label
                        "sample": {
                            "feature": {
                                **features[12],
                                "attention_matrix": features[12]["attention_matrix"].tolist()
                            },
                            "label": labels[12]
                        }
                    }
                    with open(os.path.join(METADATA_DIR, f"{filename[:-4]}.json"), 'w') as f:
                        json.dump(metadata, f, indent=2)

                # memory release
                del outputs
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM at {param} qa{qa_idx}, skipping...")
                    continue
                else:
                    raise
                
            pbar.update(1)
    
    pbar.close()

# ======================
# use generate_dataset
# ======================
ds_generator(model, tokenizer, data, device)
print("Dataset generation completed.")