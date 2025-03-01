# For db
import json
import torch
import re
import os
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, utils
from datasets import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

utils.logging.set_verbosity_error()  # Suppress standard warnings

models = [
        # 'meta-llama/Llama-3.2-1B', # Llama-3.2-1B (16 layers 32 heads)
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

SPECIAL_LABEL = -1
CACHE_DIR = "./.cache/huggingface/datasets"
FEATURE_DIR = "./training_data/features"
METADATA_DIR = "./training_data/metadata"
DS_NAME = "yahma/alpaca-cleaned"

temperatures = [0.1, 0.3, 0.5, 0.9] # low, mid, high creativity
top_k_values = [1, 3, 5, 10] # low, mid, high diversity
repetition_penalties = [1.0, 1.3, 1.6] # low, mid, high coherence
max_new_tokens_values =[100, 300, 500] # low, mid, high length

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


# Select model
print("Choose the model to test:")
for i, model in enumerate(models, 1):
    print(f" {i}. {model.split('/')[-1]}")
model_choice = int(input(f"Enter model number (1-{len(models)}): "))
if model_choice < 1 or model_choice > len(models):
    raise ValueError(f"Invalid model choice. Please enter a number between 1 and {len(models)}")
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
