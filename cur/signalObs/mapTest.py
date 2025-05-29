import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

utils.logging.set_verbosity_error()  # Suppress standard warnings
# Load tokenizer and model
models = [
        'meta-llama/Llama-3.2-1B', # Llama-3.2-1B (16 layers 32 heads)
        'meta-llama/Meta-Llama-3-8B', # Meta-Llama-3-8B
        'meta-llama/Meta-Llama-3-70B', # Meta-Llama-3-70B
        'meta-llama/Meta-Llama-3-8B-Instruct', # Meta-Llama-3-8B-Instruct (the only model TRAIL tests with)
        'EleutherAI/gpt-j-6b', # GPT-J 
        'meta-llama/Llama-2-13b-chat-hf', # Llama-2, fine-tuned
        'EleutherAI/gpt-neox-20b', # GPT-NeoX
        ]

print("Choose the model to test:\n 1. Llama-3.2-1B\n 2. Meta-Llama-3-8B\n 3. Meta-Llama-3-70B\n 4. Meta-Llama-3-8B-Instruct\n 5. GPT-J\n 6. Llama-2\n 7. GPT-NeoX")
model_choice = int(input("Enter the number corresponding to the model: "))
if model_choice < 1 or model_choice > len(models):
    raise ValueError("Invalid model choice. Please enter a number between 1 and 7.")
model_name = models[model_choice - 1]

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/research/d2/gds/hdlu24/cache_model", device_map="auto", attn_implementation="eager")
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/research/d2/gds/hdlu24/cache_model", attn_implementation="eager").to(device)

# Parameters
eos_token_id = tokenizer.eos_token_id
eos_flag=0

# Load the dataset
# alpaca dataset
# with open('./data/dataset_alpaca.json', 'r') as f:
# with open('./data/datasetSimplified_alpaca.json', 'r') as f:
datasets = {
    1: '../data/dataset_alpaca.json',
    2: '../data/datasetSimplified_alpaca.json',
    3: '../data/dataset_lmsys-chat-1m.json'
}

print("Choose the dataset to use:\n 1. Alpaca\n 2. Simplified Alpaca\n 3. LMSYS Chat 1M")
dataset_choice = int(input("Enter the number corresponding to the dataset: "))
if dataset_choice < 1 or dataset_choice > len(datasets):
    raise ValueError("Invalid dataset choice. Please enter a number between 1 and 3.")
dataset_path = datasets[dataset_choice]

with open(dataset_path, 'r') as f:
    data = json.load(f)

prompt_id = int(input(f"Enter the prompt ID (0 to {len(data['qa_pairs']) - 1}): "))
if prompt_id < 0 or prompt_id >= len(data['qa_pairs']):
    raise ValueError(f"Invalid prompt ID. Please enter a number between 0 and {len(data['qa_pairs']) - 1}.")

prompt = data['qa_pairs'][prompt_id]['prompt']

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors='pt')
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
input_len = input_ids.shape[1]

print(f"Selected prompt: {prompt}")

# Parameters
max_new_tokens = 300

# system parameters
temperature = 1  # Lower temperature for more deterministic output
top_k = 1  # Increase top_k for more diverse candidates
repetition_penalty = 1.3  # Increase repetition penalty to reduce repetition

# Initialize variables
original_input_length = input_ids.shape[1]
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

# Create directory to store attention scores
scores_dir = f'../.attention_scores/prompt_full_{prompt_id}/'
os.makedirs(scores_dir, exist_ok=True)

# Initialize lists to collect attention scores
attention_scores_all = {
    (layer_idx, head_idx): [] for layer_idx in range(num_layers) for head_idx in range(num_heads)
}

# Initialize list to collect generated tokens
generated_tokens = []
previous_token = None  # To track repetitive tokens

# Function for top-k sampling
def sample_top_k(logits, k, temperature):
    logits = logits / temperature
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze()
    return top_k_indices[0, next_token]

torch.cuda.empty_cache()

# Token generation loop
step = 0
while step < max_new_tokens:
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    logits = outputs.logits[:, -1, :]
    
    # Apply repetition penalty
    for token in generated_tokens:
        logits[0, token] /= repetition_penalty  # Adjust for batch dimension
    
    # Perform top-k sampling
    next_token = sample_top_k(logits, k=top_k, temperature=temperature)
    
    if next_token.item() == eos_token_id:
        eos_flag=1
        print("********************************")
        print("Generation ended with EOS token.")
        print("********************************")
        break
    
    # Avoid immediate repetitive tokens
    if next_token.item() == previous_token:
        logits[0, next_token.item()] = -float('inf')
        next_token = sample_top_k(logits, k=top_k, temperature=temperature)
    previous_token = next_token.item()
    
    # Collect attention scores for the last token in the sequence for each layer and head
    for layer_idx in range(num_layers):
        attentions = outputs.attentions[layer_idx]
        attention_scores = attentions[0, :, -1, :].detach().cpu().numpy()  # Shape: (num_heads, seq_len)
        for head_idx in range(num_heads):
            seq_len = original_input_length + step + 1
            scores = attention_scores[head_idx, :seq_len]  # Get scores up to current step
            attention_scores_all[(layer_idx, head_idx)].append(scores)
    
    # Append the next token to input_ids and attention_mask
    input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(device)], dim=1)
    generated_tokens.append(next_token.item())
    step += 1



# Calculate the total number of tokens generated
total_tokens = len(generated_tokens)
# Decode and print the generated output
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Ensure the output directory exists for plots
output_dir = f'../images/maps_final/{model_name}_{dataset_choice}_prompt_{prompt_id}/'
os.makedirs(output_dir, exist_ok=True)

# Create a .txt file to save the prompt, generated output, and output length info
txt_file_path = os.path.join(output_dir, 'generation_info.txt')
with open(txt_file_path, 'w') as f:
    f.write(f"Prompt: {prompt}\n")
    f.write(f"Generated output: {generated_text}\n")
    f.write(f"Total tokens generated: {total_tokens}\n")
    if eos_flag==1:
        f.write("Generation ended with EOS token.")

# Plotting attention scores as heatmaps
for (layer_idx, head_idx), scores_list in attention_scores_all.items():
    if not scores_list:
        continue  # Skip empty lists
    
    # Find the maximum sequence length
    max_seq_len = max(len(score) for score in scores_list)
    
    # Pad each score to max_seq_len
    padded_scores = [np.pad(score, (0, max_seq_len - len(score)), mode='constant') for score in scores_list]
    
    # Stack the padded scores horizontally
    heatmap_data = np.column_stack(padded_scores)
    
    # Normalize per generation step
    heatmap_data_normalized = heatmap_data / np.max(heatmap_data, axis=0, keepdims=True)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    plt.imshow(heatmap_data_normalized, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    plt.xlabel('Generation Step')
    plt.ylabel('Sequence Position')
    plt.title(f'Layer {layer_idx}, Head {head_idx}')
    plt.colorbar(label='Attention Score')
    
    # Set ticks at intervals
    x_ticks = np.arange(0, heatmap_data_normalized.shape[1], max(1, total_tokens // 50))
    y_ticks = np.arange(0, heatmap_data_normalized.shape[0], max(1, total_tokens // 50))
    '''
    if x_ticks[-1] != heatmap_data_normalized.shape[1]-1:
        x_ticks = np.append(x_ticks, heatmap_data_normalized.shape[1]-1)
    if y_ticks[-1] != heatmap_data_normalized.shape[0]-1:
        y_ticks = np.append(y_ticks, heatmap_data_normalized.shape[0]-1)
    '''
    plt.xticks(ticks=x_ticks, labels=x_ticks + 1, fontsize=6)
    plt.yticks(ticks=y_ticks, labels=y_ticks + 1, fontsize=6)
    
    # Save the plot
    plt.savefig(f'{output_dir}heatmap_layer_{layer_idx}_head_{head_idx}.png', bbox_inches='tight')
    plt.close()
