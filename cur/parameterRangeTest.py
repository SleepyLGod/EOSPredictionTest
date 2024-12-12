import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, utils
from torch.nn import functional as F

utils.logging.set_verbosity_error()  # Suppress standard warnings
# Load tokenizer and model
models = [
        'meta-llama/Llama-3.2-1B', # Llama-3.2-1B
        'meta-llama/Meta-Llama-3-8B', # Meta-Llama-3-8B
        'meta-llama/Meta-Llama-3-70B', # Meta-Llama-3-70B
        'meta-llama/Meta-Llama-3-8B-Instruct', # Meta-Llama-3-8B-Instruct
        'EleutherAI/gpt-j-6', # GPT-J 
        'meta-llama/Llama-2-13b', # Llama-2
        'EleutherAI/gpt-neox-20b', # GPT-NeoX
        ]
model_name = models[0]
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)

# Parameters
eos_token_id = tokenizer.eos_token_id

# Load the dataset
# alpaca dataset
# with open('./data/dataset_alpaca.json', 'r') as f:
with open('./data/datasetSimplified_alpaca.json', 'r') as f:
    data = json.load(f)
prompt_id = 3
prompt = data['qa_pairs'][prompt_id]['prompt']

# Create directory to store attention scores
scores_dir = f'./.attention_scores/prompt_full_{prompt_id}/'
os.makedirs(scores_dir, exist_ok=True)

print(f"Prompt: {prompt}")

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors='pt')
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
input_len = input_ids.shape[1]

print(f"Input length: {input_len}")
# print(f"Input tokens: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
print(f"Attention mask: {attention_mask}")

# other parameters

initial_input_ids = input_ids
initial_attention_mask = torch.ones_like(initial_input_ids)

# Function for top-k sampling
def sample_top_k(logits, k, temperature):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze()
    return top_k_indices[0, next_token]

# General function to generate text, record output lengths, and plot/save graphs
def generate_and_record(varying_param_name, varying_param_values, fixed_params, output_prefix):
    output_lengths = []
    for param_value in varying_param_values:
        current_params = fixed_params.copy()
        current_params[varying_param_name] = param_value
        input_ids = initial_input_ids.clone()
        attention_mask = initial_attention_mask.clone()
        generated_tokens = []
        step = 0
        max_new_tokens = 400
        while step < max_new_tokens:
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            
            # Apply repetition penalty
            for token in generated_tokens:
                logits[0, token] /= current_params['repetition_penalty']
            
            # Avoid NaN and Inf in logits
            logits = torch.where(torch.isnan(logits), torch.full_like(logits, -float('inf')), logits)
            logits = torch.where(torch.isinf(logits), torch.full_like(logits, -float('inf')), logits)
            
            # Perform top-k sampling
            next_token = sample_top_k(logits, k=current_params['top_k'], temperature=current_params['temperature'])
            
            if next_token.item() == eos_token_id:
                break
            
            # Avoid immediate repetitive tokens
            if generated_tokens and next_token.item() == generated_tokens[-1]:
                logits[0, next_token.item()] = -float('inf')
                next_token = sample_top_k(logits, k=current_params['top_k'], temperature=current_params['temperature'])
            
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
            generated_tokens.append(next_token.item())
            step += 1
        
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        total_tokens = len(generated_tokens)
        output_lengths.append(total_tokens)
        
        # Save to .txt file
        os.makedirs(os.path.dirname(f'.output/{output_prefix}_params_{param_value}.txt'), exist_ok=True)
        with open(f'.output/{output_prefix}_params_{param_value}.txt', 'w') as f:
            f.write(f"Parameters: {current_params}\n")
            f.write(f"Generated output: {generated_text}\n")
            f.write(f"Total tokens generated: {total_tokens}\n\n")
        print(f"Tested {varying_param_name} = {param_value}, generated {total_tokens} tokens")

    # Plot and save graph
    plt.figure()
    plt.plot(varying_param_values, output_lengths, marker='o')
    plt.xlabel(varying_param_name)
    plt.ylabel('output_length')
    print(f"Tested {varying_param_name} = {param_value}, generated {total_tokens} tokens")
    plt.title(f'output_length & {varying_param_name} for prompt {prompt_id}')
    plt.grid(True)
    plt.savefig(f'.output/{output_prefix}_graph.png')
    plt.close()

# Experiment 1: Vary repetition_penalty
repetition_penalties = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6]
fixed_params = {'temperature': 0.5, 'top_k': 1}
generate_and_record('repetition_penalty', repetition_penalties, fixed_params, f'prompt_{prompt_id}/exp1')

# Experiment 2: Vary temperature
temperatures = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
fixed_params = {'repetition_penalty': 1.3, 'top_k': 5}
generate_and_record('temperature', temperatures, fixed_params, f'prompt_{prompt_id}/exp2')

# Experiment 3: Vary top_k
top_ks = [1, 6, 11, 16, 21, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
fixed_params = {'repetition_penalty': 1.3, 'temperature': 0.5}
generate_and_record('top_k', top_ks, fixed_params, f'prompt_{prompt_id}/exp3')
