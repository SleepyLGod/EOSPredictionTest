import json
import torch
import os
import re
import math
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
        
        "chavinlo/gpt4-x-alpaca", # GPT4-X-ALPACA
        "TurkuNLP/gpt3-finnish-large", # GPT3-Finnish-Large
        "TurkuNLP/gpt3-finnish-xl", # GPT3-Finnish-XL
        "CreitinGameplays/gpt3-finnish-xl-alpaca", # GPT3-Finnish-XL-ALPACA
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # DeepSeek-R1-Distill-Qwen-1.5B
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # DeepSeek-R1-Distill-Llama-8B
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", # DeepSeek-R1-Distill-Qwen-14B
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # DeepSeek-R1-Distill-Qwen-32B
        
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

temperatures = [0.1, 0.5, 0.9] # low, mid, high creativity
top_k_values = [1, 2, 3, 10] # low, mid, high diversity
repetition_penalties = [1.0, 1.3, 1.6] # low, mid, high coherence
max_new_tokens_values = [100, 250, 500] # low, mid, high length
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
# print(system_parameters)

TEMP_SAMP = temperatures[2]
TK_SAMP = top_k_values[3]
REP_PEN_SAMP = repetition_penalties[1]
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
for i in range(1, len(models)+1):
    model_choice = i
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


    # ADDITION_PROMPT = (
    #     "FIRST, predict the length of your response. "
    #     "Output ONLY a numerical estimate in the first line using one of these formats:\n"
    #     "- Single number: 50\n"
    #     "- Range: 40-60\n"
    #     "SECOND, write your response normally after a blank line. "
    #     "Your prediction must be in the very first line with no additional text."
    # )

    # metrics = {
    #     "accuracies": {
    #         "accuracy_1": 0,  # Error < 20%
    #         "accuracy_2": 0,  # Error < 30%
    #         "accuracy_3": 0,  # Error < 50%
    #     },
    #     "total_prompts": 0,
    #     "predicted_lengths": [],
    #     "actual_lengths": [],
    # }

    ADDITION_PROMPT = (
        "Before responding to the above instruction, you have to predict the length of your response. "
        "Print the estimated number of words in your response in the first line. "
        "Then change to a new line to respond to the instruction."
    )

    results = []
    predictions_list = []  # New list for prediction/actual pairs

    for idx in range(50):
        if idx >= len(data['qa_pairs']):
            break
            
        qa_pair = data['qa_pairs'][idx]
        original_prompt = qa_pair["prompt"]
        
        if ADDITION_PROMPT not in qa_pair["prompt"]:
            qa_pair["prompt"] = f"{qa_pair['prompt']}\n{ADDITION_PROMPT}"
            
        entry = {
            "original_prompt": qa_pair["prompt"],
            # "dataset_response": qa_pair["response"],
            "model_response": "",
            "exceeded_max_seq_len": False,
            "total_sequence_length": 0  # Now represents output tokens (total - input)
        }
        
        # Tokenize prompt
        inputs = tokenizer(qa_pair["prompt"], return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        input_token_count = input_ids.shape[1]  # Store input length
        
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
        # eos_reached = False
        
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
                # eos_reached = True
                break
                
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(device)], dim=1)
            current_length += 1
        
        output_token_count = current_length - input_token_count  # Calculate output length
        entry["model_response"] = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        entry["exceeded_max_seq_len"] = exceeded or (current_length > max_seq_len)
        entry["total_sequence_length"] = output_token_count
        
        # Extract prediction from first line
        prediction_line = entry["model_response"].split("\n", 1)[0].strip()
        predictions_list.append({
            "model output length prediction": prediction_line,
            "actual output length": output_token_count
        })

        results.append(entry)
        print(f"Processed prompt {idx+1}/100")

    # Create final output structure
    output_data = {
        "predictions": predictions_list,
        "detailed_results": results
    }

    # Save results
    preview_dir = "./preview/pe/change2/"
    os.makedirs(preview_dir, exist_ok=True)

    # Save results in the preview directory
    output_file = os.path.join(preview_dir, f"{model_name.split('/')[-1]}_responses_pe.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {output_file}")
    torch.cuda.empty_cache()
    print("Cleared CUDA cache\n")
    print("--------------------------------------------------\n")