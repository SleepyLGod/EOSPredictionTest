import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import os
import random
import json # For saving detailed results
from datetime import timedelta # For timeout, if needed anywhere else
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from datasets import load_dataset # use a Hugging Face dataset for prompts

# <editor-fold desc="Helper classes and functions from training/previous script">
HIDDEN_SIZE = 8192
MIN_TEMP, MAX_TEMP = 0.1, 0.9
MIN_TOP_K, MAX_TOP_K = 0, 127.0 # Assuming 0 is a valid unnormalized top_k after decode
MIN_REP_PENALTY, MAX_REP_PENALTY = 1.3, 1.6
MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS = 0, 700.0 # Or 300-500 if fixed
MIN_SEQ_POS, MAX_SEQ_POS = 0, 4095.0 # Or based on LLM max length

# 原始 decode_params 函数（用于从编码值解码，测试时我们直接用原始参数）
# def decode_params_from_training(encoded): ...

# 将原始解码参数归一化，就像训练时 Dataset 做的那样
def normalize_eval_params(temp, top_k, rep_p, max_len_val, seq_pos_val):
    def _normalize(v, min_v, max_v):
        if max_v == min_v: return 0.0 if v == min_v else 0.5
        return np.clip((v - min_v) / (max_v - min_v), 0.0, 1.0)

    return {
        'temperature': np.float32(_normalize(temp, MIN_TEMP, MAX_TEMP)),
        'top_k': np.float32(_normalize(top_k, MIN_TOP_K, MAX_TOP_K)),
        'repetition_penalty': np.float32(_normalize(rep_p, MIN_REP_PENALTY, MAX_REP_PENALTY)),
        'max_len': np.float32(_normalize(max_len_val, MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS)),
        'seq_pos': np.float32(_normalize(seq_pos_val, MIN_SEQ_POS, MAX_SEQ_POS)),
    }

class EnhancedMLP(torch.nn.Module): # Use your actual trained MLP definition
    def __init__(self, hidden_size):
        super().__init__()
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + 5, 2048), torch.nn.BatchNorm1d(2048), torch.nn.GELU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(2048, 1024), torch.nn.LayerNorm(1024), torch.nn.GELU()
        )
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(1024, 512), torch.nn.SiLU(), torch.nn.Linear(512, 1)
        )
        self.residual = torch.nn.Linear(hidden_size + 5, 1024)
    def forward(self, x_dict): # x_dict contains 'embedding' and normalized param tensors
        params_to_cat = [x_dict['embedding']]
        # Ensure params are [batch_size, 1]
        for param_name in ['temperature', 'top_k', 'repetition_penalty', 'max_len', 'seq_pos']:
            tensor = x_dict[param_name]
            if tensor.dim() == 1: tensor = tensor.unsqueeze(1) # if batch_size=1
            elif tensor.dim() == 0: tensor = tensor.unsqueeze(0).unsqueeze(1) # if scalar
            params_to_cat.append(tensor)
        main_feature = torch.cat(params_to_cat, dim=1)
        interacted = self.interaction(main_feature)
        residual_out = self.residual(main_feature)
        fused = interacted + residual_out
        return self.reg_head(fused).squeeze(-1)
# </editor-fold>

# --- 配置 ---
LLM_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B' # LLM to drive the generation
LENGTH_PREDICTOR_PATH = Path("path/to/your/saved_models_ddp_robust_final/your_run_timestamp/enhanced_mlp_best.pth") # <--- 修改
PROMPT_SOURCE_FILE = None # Path to a .json file with prompts, or use a default list
# Example prompt structure in JSON: [{"id": "prompt1", "text": "Translate to French: Hello world."}, ...]
NUM_TEST_PROMPTS = 10 # Number of prompts to test from the source
OUTPUT_RESULTS_FILE = Path(f"./length_predictor_eval_results_{LLM_MODEL_NAME.split('/')[-1]}.json")
DEVICE_LLM = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Device for LLM
DEVICE_PREDICTOR = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_LLM) # Device for your length predictor (can be same or different)

# Decoding parameters to test (list of dicts)
DECODING_PARAMS_LIST = [
    {'temperature': 0.7, 'top_k': 50, 'repetition_penalty': 1.0, 'max_new_tokens': 100},
    {'temperature': 0.1, 'top_k': 1, 'repetition_penalty': 1.2, 'max_new_tokens': 200},
    # Add more combinations
]

def load_prompts(source_file, num_prompts):
    if source_file and Path(source_file).exists():
        with open(source_file, 'r') as f:
            all_prompts_data = json.load(f) # Expects a list of dicts with "text" key
        return all_prompts_data[:num_prompts]
    else:
        # Default prompts if no file provided
        return [
            {"id": "default_1", "text": "Translate the following English text to French: 'Hello, how are you today?'"},
            {"id": "default_2", "text": "Write a short story about a robot who dreams of becoming a painter."},
            {"id": "default_3", "text": "What is the capital of Australia?"},
        ][:num_prompts]


def sample_next_token(logits, temperature=1.0, top_k=0):
    """ Samples a token from logits with temperature and top-k filtering. """
    if temperature > 0:
        logits = logits / temperature
    
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        # Create a mask for all other tokens
        filter_value = -float('Inf')
        mask = torch.ones_like(logits, dtype=torch.bool)
        # This part needs to be careful to only keep top_k on the original vocab positions
        # A simpler way for sampling:
        # Get probabilities of top_k tokens
        probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
        # Sample from the top_k tokens
        sampled_relative_idx = torch.multinomial(probs, 1)
        return top_k_indices.gather(-1, sampled_relative_idx)
    else: # Greedy or multinomial from full distribution
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)


@torch.no_grad()
def evaluate_length_predictor():
    print(f"Loading LLM: {LLM_MODEL_NAME} on {DEVICE_LLM}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token 
        print(f"Set LLM pad_token to eos_token: {llm_tokenizer.eos_token}")

    try:
        # Use flash attention if available, requires appropriate PyTorch and CUDA versions
        attn_impl = "flash_attention_2" if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16 if DEVICE_LLM.type == 'cuda' else torch.float32, # bfloat16 for CUDA
            device_map=DEVICE_LLM if DEVICE_LLM.type == 'cuda' else None, # device_map for single device
            attn_implementation=attn_impl,
            output_hidden_states=True # Crucial for extracting embeddings
        )
        if DEVICE_LLM.type == 'cpu' and attn_impl == "flash_attention_2": # FA2 not for CPU
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, torch_dtype=torch.float32, output_hidden_states=True
            ).to(DEVICE_LLM)

    except Exception as e:
        print(f"Error loading LLM with potential flash_attention_2: {e}")
        print("Falling back to standard attention.")
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16 if DEVICE_LLM.type == 'cuda' else torch.float32,
            device_map=DEVICE_LLM if DEVICE_LLM.type == 'cuda' else None,
            attn_implementation="eager",
            output_hidden_states=True
        ).to(DEVICE_LLM)
    llm_model.eval()

    print(f"Loading Length Predictor from: {LENGTH_PREDICTOR_PATH} on {DEVICE_PREDICTOR}")
    length_predictor = EnhancedMLP(HIDDEN_SIZE).to(DEVICE_PREDICTOR)
    try:
        state_dict = torch.load(LENGTH_PREDICTOR_PATH, map_location=DEVICE_PREDICTOR)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_ddp_model = any(key.startswith('module.') for key in state_dict.keys())
        if is_ddp_model:
            for k, v in state_dict.items(): new_state_dict[k[7:]] = v
            length_predictor.load_state_dict(new_state_dict)
        else:
            length_predictor.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"ERROR: Length predictor model file not found at {LENGTH_PREDICTOR_PATH}")
        return
    except Exception as e:
        print(f"Error loading length predictor state_dict: {e}. Trying as checkpoint.")
        try:
            checkpoint = torch.load(LENGTH_PREDICTOR_PATH, map_location=DEVICE_PREDICTOR)
            state_dict = checkpoint['model_state_dict']
            new_state_dict = OrderedDict()
            is_ddp_model = any(key.startswith('module.') for key in state_dict.keys())
            if is_ddp_model:
                for k, v in state_dict.items(): new_state_dict[k[7:]] = v
                length_predictor.load_state_dict(new_state_dict)
            else:
                length_predictor.load_state_dict(state_dict)
            print(f"Loaded length predictor from checkpoint (epoch {checkpoint.get('epoch', 'N/A')}).")
        except Exception as e_chk:
            print(f"ERROR: Could not load length predictor: {e_chk}")
            return
    length_predictor.eval()

    prompts_data = load_prompts(PROMPT_SOURCE_FILE, NUM_TEST_PROMPTS)
    all_results = []

    for prompt_info in tqdm(prompts_data, desc="Processing Prompts"):
        prompt_text = prompt_info["text"]
        prompt_id = prompt_info.get("id", str(random.randint(1000,9999)))

        for dec_params_idx, dec_params in enumerate(DECODING_PARAMS_LIST):
            current_max_new_tokens = dec_params['max_new_tokens']
            
            inputs = llm_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_POS - current_max_new_tokens - 5).to(DEVICE_LLM) # Ensure space for generation
            input_ids = inputs.input_ids
            original_prompt_len = input_ids.shape[1]
            
            if original_prompt_len == 0:
                print(f"Warning: Prompt '{prompt_id}' resulted in empty input_ids, skipping.")
                continue

            generated_ids_list = [] # Store all generated token_ids for this prompt & param_set
            past_key_values = None
            prompt_decoding_results = []

            for step in range(current_max_new_tokens):
                current_seq_len = original_prompt_len + step # Length *before* generating current token
                
                # 1. Get features for length predictor
                llm_outputs = llm_model(
                    input_ids if step == 0 else input_ids[:, -1:], # Only feed last token after first step
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Embedding is from the token that *will be used to predict the next token*
                last_token_embedding = llm_outputs.hidden_states[-1][:, -1, :].to(DEVICE_PREDICTOR) # [1, HIDDEN_SIZE]
                
                # Normalize parameters for length predictor input
                norm_params = normalize_eval_params(
                    dec_params['temperature'], dec_params['top_k'], dec_params['repetition_penalty'],
                    dec_params['max_new_tokens'], # This is the overall max_new_tokens for the generation session
                    current_seq_len # current position in sequence
                )

                predictor_input = {
                    'embedding': last_token_embedding.to(torch.float32), # Ensure float32 for MLP
                    'temperature': torch.tensor([norm_params['temperature']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    'top_k': torch.tensor([norm_params['top_k']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    'repetition_penalty': torch.tensor([norm_params['repetition_penalty']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    'max_len': torch.tensor([norm_params['max_len']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    'seq_pos': torch.tensor([norm_params['seq_pos']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                }
                
                # 2. Length Prediction
                start_time = time.perf_counter()
                if DEVICE_PREDICTOR.type == 'cuda': torch.cuda.synchronize(DEVICE_PREDICTOR)
                predicted_rest_len_tensor = length_predictor(predictor_input)
                if DEVICE_PREDICTOR.type == 'cuda': torch.cuda.synchronize(DEVICE_PREDICTOR)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                predicted_rest_len = predicted_rest_len_tensor.item() # Get scalar value

                # 3. LLM generates next token (for continuing the simulation)
                logits = llm_outputs.logits[:, -1, :] # Logits for the next token
                past_key_values = llm_outputs.past_key_values

                # Apply repetition penalty manually if needed (or ensure LLM handles it)
                # Note: Your training data generation applied it after getting logits.
                # For consistency, you might do the same here if LLM itself doesn't handle it in .generate()
                # For this simulation, we'll sample from the direct logits.
                
                # Simple sampling (replace with your actual sampling if more complex)
                # next_token_id = torch.argmax(logits, dim=-1, keepdim=True) # Greedy
                next_token_id = sample_next_token(logits, dec_params['temperature'], dec_params['top_k'])
                if not isinstance(next_token_id, torch.Tensor) or next_token_id.ndim == 0:
                    next_token_id = torch.tensor([[next_token_id]], device=DEVICE_LLM)
                elif next_token_id.ndim == 1 and next_token_id.numel() == 1:
                    next_token_id = next_token_id.view(1,1)
                
                generated_ids_list.append(next_token_id.item())

                # 4. Calculate actual remaining length for this step
                # This is tricky: actual_rest_len depends on when EOS *would* be generated by the LLM.
                # For simplicity in this eval, we can define "actual" based on current max_new_tokens.
                # If you have ground truth full responses, you could use that.
                actual_rest_len = current_max_new_tokens - (step + 1) # +1 because step is 0-indexed for generated tokens

                prompt_decoding_results.append({
                    "step": step,
                    "current_seq_len": current_seq_len,
                    "predicted_rest_len": predicted_rest_len,
                    "actual_rest_len": actual_rest_len, # Based on max_new_tokens limit for this step
                    "prediction_error": predicted_rest_len - actual_rest_len,
                    "latency_ms": latency_ms,
                    "generated_token_id": next_token_id.item(),
                })

                # Update input for next LLM step
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == llm_tokenizer.eos_token_id:
                    # print(f"EOS token encountered at step {step} for prompt {prompt_id}")
                    break # Stop generation for this prompt

            all_results.append({
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "decoding_params_idx": dec_params_idx,
                "decoding_params": dec_params,
                "original_prompt_len": original_prompt_len,
                "generated_tokens_count": len(generated_ids_list),
                "generated_token_ids": generated_ids_list,
                "step_predictions": prompt_decoding_results
            })
            
            # Clean up for next param set or prompt
            del past_key_values
            if DEVICE_LLM.type == 'cuda': torch.cuda.empty_cache()


    # Save all detailed results
    with open(OUTPUT_RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {OUTPUT_RESULTS_FILE}")

    # --- Aggregate Statistics ---
    print("\n--- Aggregate Statistics ---")
    all_prediction_errors = []
    all_latencies = []
    for res_group in all_results:
        for step_res in res_group["step_predictions"]:
            all_prediction_errors.append(step_res["prediction_error"])
            all_latencies.append(step_res["latency_ms"])

    if all_prediction_errors:
        all_prediction_errors = np.array(all_prediction_errors)
        mae = np.mean(np.abs(all_prediction_errors))
        mse = np.mean(all_prediction_errors**2)
        rmse = np.sqrt(mse)
        bias = np.mean(all_prediction_errors) # Positive means over-prediction, negative under-prediction
        
        print(f"Overall Prediction MAE: {mae:.2f} tokens")
        print(f"Overall Prediction RMSE: {rmse:.2f} tokens")
        print(f"Overall Prediction Bias: {bias:.2f} tokens")
        # You can also calculate these per decoding_params set or per prompt length bucket
    else:
        print("No prediction errors recorded.")

    if all_latencies:
        all_latencies = np.array(all_latencies)
        avg_latency = np.mean(all_latencies)
        median_latency = np.median(all_latencies)
        p90_latency = np.percentile(all_latencies, 90)
        p95_latency = np.percentile(all_latencies, 95)
        p99_latency = np.percentile(all_latencies, 99)
        
        print(f"\nPredictor Average Latency: {avg_latency:.2f} ms")
        print(f"Predictor Median Latency: {median_latency:.2f} ms")
        print(f"Predictor P90 Latency: {p90_latency:.2f} ms")
        print(f"Predictor P95 Latency: {p95_latency:.2f} ms")
        print(f"Predictor P99 Latency: {p99_latency:.2f} ms")
    else:
        print("No latencies recorded.")
    print("--------------------------")


if __name__ == "__main__":
    # Ensure paths are correctly set by the user
    if "path/to/your" in str(LENGTH_PREDICTOR_PATH):
        print(f"ERROR: LENGTH_PREDICTOR_PATH is a placeholder: {LENGTH_PREDICTOR_PATH}")
        print("Please update it to the actual path of your trained .pth file.")
    elif not LENGTH_PREDICTOR_PATH.exists():
        print(f"ERROR: LENGTH_PREDICTOR_PATH does not exist: {LENGTH_PREDICTOR_PATH}")
    else:
        evaluate_length_predictor()