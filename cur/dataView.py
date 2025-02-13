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
CACHE_DIR = "~/.cache/huggingface/datasets"
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

# Generate responses
results = []
for idx in range(100):
    if idx >= len(data['qa_pairs']):
        break
        
    qa_pair = data['qa_pairs'][idx]
    entry = {
        "original_prompt": qa_pair["prompt"],
        "dataset_response": qa_pair["response"],
        "model_response": "",
        "exceeded_max_seq_len": False,
        "total_sequence_length": 0
    }
    
    # Tokenize prompt
    inputs = tokenizer(qa_pair["prompt"], return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Check input length
    if input_ids.shape[1] > max_seq_len:
        entry["exceeded_max_seq_len"] = True
        entry["total_sequence_length"] = input_ids.shape[1]
        results.append(entry)
        continue
    
    # Generate response
    generated_tokens = []
    current_length = input_ids.shape[1]
    exceeded = False
    eos_reached = False
    
    for _ in range(max_new_tokens):
        if current_length >= max_seq_len:
            exceeded = True
            break
            
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits[:, -1, :]
        
        # Apply repetition penalty
        for token in generated_tokens:
            logits[0, token] /= repetition_penalty
            
        next_token = sample_top_k(logits, top_k, temperature)
        
        if next_token == eos_token_id:
            eos_reached = True
            break
            
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(device)], dim=1)
        current_length += 1
    
    entry["model_response"] = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    entry["exceeded_max_seq_len"] = exceeded or (current_length > max_seq_len)
    entry["total_sequence_length"] = current_length
    results.append(entry)
    
    print(f"Processed prompt {idx+1}/100")

# Save results
output_file = f"{model_name.split('/')[-1]}_responses.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")