import json
import torch
import os
import math
import numpy as np
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
        ]

datasets = {
        1: './data/dataset_alpaca.json',
        2: './data/datasetSimplified_alpaca.json',
        3: './data/dataset_lmsys-chat-1m.json',
        4: 'yahma/alpaca-cleaned'
}

SPECIAL_LABEL = -1
CACHE_DIR = "./.cache/huggingface/datasets"
FEATURE_DIR = "./training_data/features"
METADATA_DIR = "./training_data/metadata"
DS_NAME = datasets[4]

# Parameters
max_new_tokens = 300
temperature = 1  # Lower temperature for more deterministic output
top_k = 1  # Increase top_k for more diverse candidates
repetition_penalty = 1.3  # Increase repetition penalty to reduce repetition

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
print("Choose the model to test:")
for i, model in enumerate(models, 1):
    print(f" {i}. {model.split('/')[-1]}")
model_choice = int(input("Enter model number (1-7): "))
if model_choice < 1 or model_choice > len(models):
    raise ValueError("Invalid model choice. Please enter a number between 1 and 7.")
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
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    return top_k_indices[0, torch.multinomial(probs, 1).item()]

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

num_layers = model.config.num_layers
num_heads = model.config.num_heads
print(f"Number of layers: {num_layers}")
print(f"Number of heads: {num_heads}")

# --- traverse dataset ---
for qa_idx, qa in enumerate(data['qa_pairs']):
    prompt = qa['prompt']
    target_response = qa.get('response', '')

    # --- initialize generation state ---
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    generated_tokens = []
    eos_encountered = False
    features = []
    seq_poses = []

    # --- generate tokens ---
    step = 0
    while step < max_new_tokens:
        try:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM at prompt {qa_idx}, skipping...")
                break
            else:
                raise
        
        # --- feature collection ---
        # input_ids.shape[0]: batch size, input_ids.shape[1]: sequence length, input_ids.shape[2]: vocab size
        # outputs.attentions: list of length num_layers, 
        #                     each element is a tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
        current_seq_pos = input_ids.shape[1] - 1 # position before the new token
        seq_poses.append(current_seq_pos)

        # attention weights substraction
        layer_head_weights = []
        for layer_idx in range(num_layers):
            attentions = outputs.attentions[layer_idx] # (1, num_heads, sequence_length, sequence_length)
            head_weights = attentions[0, :, -1, :].cpu().numpy() # (num_heads, sequence_length)
            layer_head_weights.append(head_weights)

        # transfer to 3D numpy array (num_layers, num_heads, sequence_length)
        feature_step = np.stack(layer_head_weights, axis=0) # (num_layers, num_heads, sequence_length)
        features.append(feature_step)

        # --- token generation ---
        logits = outputs.logits[:, -1, :]
        # Apply repetition penalty
        for token in generated_tokens:
            logits[0, token] /= repetition_penalty
        # sample next token
        next_token = sample_top_k(logits, top_k, temperature)

        # check if EOS token is generated
        if next_token == eos_token_id:
            eos_encountered = True
            generated_tokens.append(next_token.item())
            break

        # update input_ids and attention_mask
        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(device)], dim=1)
        generated_tokens.append(next_token.item())
        step += 1

    # --- calculate the tags ---
    total_generated_tokens = len(generated_tokens)
    lables = []
    for pos in seq_poses:
        if eos_encountered:
            remaining_tokens = max(0, total_generated_tokens - pos - 1)
        else:
            remaining_tokens = SPECIAL_LABEL
        lables.append(remaining_tokens)

    # --- save the features and metadata ---
    if len(features) > 0:
        # convert to numpy array with compressed format
        feature_array = np.stack(features, axis=0) # (num_steps, num_layers, num_heads, sequence_length)
        model_name_safe = model_name.replace('/', '_') # replace '/' with '_' in model name
        np.savez_compressed(
            os.path.join(FEATURE_DIR, f"{model_name_safe}_{qa_idx}.npz"), 
            features=feature_array.astype(np.float16),
            seq_poses=np.array(seq_poses),
            lables=np.array(lables)
        )

        # save metadata
        metadata = {
            "prompt": prompt,
            "target_response": target_response,
            "generated_response": tokenizer.decode(generated_tokens),
            "total_tokens": total_generated_tokens,
            "eos_encountered": eos_encountered,
            "exceed_max_len": total_generated_tokens >= max_new_tokens,
        }
        with open(os.path.join(METADATA_DIR, f"{model_name_safe}_{qa_idx}.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

    # release memory
    torch.cuda.empty_cache()