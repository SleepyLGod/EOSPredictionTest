import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import os
import random
import json
from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # For R2 score if needed
import sys # For exiting early
import logging # For better logging control

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# <editor-fold desc="Helper classes and functions from training/previous script">
# --- 常量与配置 (与训练时一致或根据需要调整) ---
HIDDEN_SIZE = 8192 # 必须与训练时模型一致

# 数据归一化参数 (必须与训练时完全一致!)
# 确保这些值覆盖了 DECODING_PARAMS_LIST 中使用的所有参数
MIN_TEMP, MAX_TEMP = 0.1, 0.9
MIN_TOP_K, MAX_TOP_K = 1, 100 # 假设 top_k >= 1 (如果为0表示禁用，则需要调整归一化或逻辑)
MIN_REP_PENALTY, MAX_REP_PENALTY = 1.0, 1.6 # 调整下限以包含测试参数
MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS = 100, 500 # 调整以匹配测试参数范围
MIN_SEQ_POS, MAX_SEQ_POS = 0, 8191 # 例如 Llama 3 8k 上下文 (实际应从LLM配置获取)

# --- 配置 ---
LLM_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
LENGTH_PREDICTOR_PATH = Path("./saved_models_ddp_robust_final/your_run_timestamp/enhanced_mlp_best.pth") # <--- 修改
PROMPT_SOURCE_FILE = None # Path to a .json file with prompts, or use a default list
NUM_TEST_PROMPTS = 5 # Reduced for quicker testing; increase as needed
OUTPUT_RESULTS_FILE_TEMPLATE = "./length_predictor_eval_results_{llm_name}_{timestamp}.json" # Added timestamp
DEVICE_LLM = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_PREDICTOR = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_LLM)
USE_FLASH_ATTENTION_2 = False # Set to False to disable Flash Attention 2

# Decoding parameters to test
# IMPORTANT: Ensure these values are within the MIN_... and MAX_... ranges defined above for normalization.
DECODING_PARAMS_LIST = [
    {'temperature': 0.7, 'top_k': 50, 'repetition_penalty': 1.0, 'max_new_tokens': 100},
    {'temperature': 0.1, 'top_k': 1, 'repetition_penalty': 1.2, 'max_new_tokens': 200},
    {'temperature': 0.9, 'top_k': 100, 'repetition_penalty': 1.5, 'max_new_tokens': 300}, # Added another example
]

# 需要一个函数来将原始解码参数归一化，就像训练时 Dataset 做的那样
def normalize_eval_params(temp, top_k_val, rep_p, max_len_session, current_seq_pos_val):
    def _normalize(v, min_v, max_v, name="param"):
        if max_v == min_v:
            # logger.debug(f"Normalization for {name}: min=max={min_v}, value={v}, returning 0.0 or 0.5")
            return 0.0 if v == min_v else 0.5
        norm_v = (v - min_v) / (max_v - min_v)
        clipped_v = np.clip(norm_v, 0.0, 1.0)
        if clipped_v != norm_v and abs(clipped_v - norm_v) > 1e-5: # If significant clipping occurs
            logger.warning(
                f"Clipping occurred for {name}: value={v} (norm={norm_v:.3f}) was outside "
                f"[{min_v}, {max_v}] and clipped to {clipped_v:.3f}. "
                f"Ensure MIN/MAX constants match DECODING_PARAMS_LIST ranges."
            )
        return clipped_v
    
    return {
        'temperature': np.float32(_normalize(temp, MIN_TEMP, MAX_TEMP, 'temperature')),
        'top_k': np.float32(_normalize(top_k_val, MIN_TOP_K, MAX_TOP_K, 'top_k')),
        'repetition_penalty': np.float32(_normalize(rep_p, MIN_REP_PENALTY, MAX_REP_PENALTY, 'repetition_penalty')),
        'max_len': np.float32(_normalize(max_len_session, MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS, 'max_len_session')), # This is the max_new_tokens for the *session*
        'seq_pos': np.float32(_normalize(current_seq_pos_val, MIN_SEQ_POS, MAX_SEQ_POS, 'seq_pos')),
    }

class EnhancedMLP(torch.nn.Module): # Use your actual trained MLP definition
    def __init__(self, hidden_size):
        super().__init__()
        # Define a default dropout rate if not specified elsewhere, or make it an arg
        mlp_dropout_rate = 0.2 # Consistent with training script's interaction block
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + 5, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.GELU(),
            torch.nn.Dropout(mlp_dropout_rate),
            torch.nn.Linear(2048, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.GELU()
        )
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 1)
        )
        self.residual = torch.nn.Linear(hidden_size + 5, 1024)

    def forward(self, x_dict):
        params_to_cat = [x_dict['embedding']]
        for param_name in ['temperature', 'top_k', 'repetition_penalty', 'max_len', 'seq_pos']:
            tensor = x_dict[param_name]
            if tensor.dim() == 0: tensor = tensor.view(1, 1) # Batch size 1, feature size 1
            elif tensor.dim() == 1: tensor = tensor.unsqueeze(1)
            params_to_cat.append(tensor)
        main_feature = torch.cat(params_to_cat, dim=1)
        interacted = self.interaction(main_feature)
        residual_out = self.residual(main_feature)
        fused = interacted + residual_out
        return self.reg_head(fused).squeeze(-1)
# </editor-fold>

def load_prompts(source_file, num_prompts):
    # ... (same as your previous version, ensure it returns list of dicts with "text" and "id") ...
    if source_file and Path(source_file).exists():
        with open(source_file, 'r') as f:
            try:
                all_prompts_data = json.load(f)
                if not isinstance(all_prompts_data, list):
                    raise ValueError("Prompt file should contain a JSON list.")
                if all_prompts_data and not isinstance(all_prompts_data[0], dict):
                    raise ValueError("Prompt list elements should be dictionaries.")
                # Ensure 'text' key exists, 'id' is optional but good for tracking
                for i, p_data in enumerate(all_prompts_data):
                    if "text" not in p_data:
                        raise ValueError(f"Prompt at index {i} missing 'text' key.")
                    if "id" not in p_data:
                        p_data["id"] = f"prompt_{i}"

            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from prompt file: {source_file}")
                return [] # Return empty or raise
            except ValueError as ve:
                logger.error(f"Error in prompt file structure: {ve}")
                return []
        selected_prompts = all_prompts_data[:num_prompts]
        logger.info(f"Loaded {len(selected_prompts)} prompts from {source_file}")
        return selected_prompts
    else:
        logger.info("No prompt source file provided or found. Using default prompts.")
        return [
            {"id": "default_translate", "text": "Translate the following English text to French: 'Hello, how are you today?'"},
            {"id": "default_story", "text": "Write a short story about a robot who dreams of becoming a painter."},
            {"id": "default_capital", "text": "What is the capital of Australia?"},
        ][:num_prompts]


def sample_next_token_from_logits(logits, temperature=1.0, top_k=0):
    """
    Samples a token from logits using temperature and top-k filtering.
    Args:
        logits: Tensor of shape (batch_size, vocab_size).
        temperature: Float for temperature scaling. If 0 or <0, treated as 1.0.
        top_k: Integer for top-k filtering. If 0, no top-k filtering (full distribution).
    Returns:
        Tensor of shape (batch_size, 1) with sampled token IDs.
    """
    if temperature <= 0: # Treat 0 or negative temp as 1.0 (no scaling)
        temperature = 1.0
    
    logits = logits / temperature

    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        # top_k_indices are the original vocab indices of the top k tokens
        # top_k_values are their corresponding logits
        
        # Sample from the top_k logits
        probs = torch.nn.functional.softmax(top_k_values, dim=-1)
        sampled_relative_indices = torch.multinomial(probs, 1) # shape (batch_size, 1)
        
        # Convert relative indices back to original vocab indices
        # .gather is used to select elements from top_k_indices using sampled_relative_indices
        return top_k_indices.gather(-1, sampled_relative_indices)
    else:
        # No top-k, sample from the full distribution
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)


@torch.no_grad()
def evaluate_length_predictor():
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_results_file = Path(
        OUTPUT_RESULTS_FILE_TEMPLATE.format(
            llm_name=LLM_MODEL_NAME.split('/')[-1],
            timestamp=run_timestamp
        )
    )
    output_results_file.parent.mkdir(parents=True, exist_ok=True)


    logger.info(f"Loading LLM: {LLM_MODEL_NAME} on {DEVICE_LLM}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        logger.info(f"Set LLM pad_token_id to eos_token_id: {llm_tokenizer.eos_token_id}")

    # Determine LLM's actual max length
    llm_model_max_length = getattr(llm_tokenizer, 'model_max_length', None)
    if llm_model_max_length is None and hasattr(llm_tokenizer, 'tokenizer_config') and 'model_max_length' in llm_tokenizer.tokenizer_config:
        llm_model_max_length = llm_tokenizer.tokenizer_config['model_max_length']
    if llm_model_max_length is None and hasattr(llm_model, 'config') and hasattr(llm_model.config, 'max_position_embeddings'):
        llm_model_max_length = llm_model.config.max_position_embeddings # Fallback
    if llm_model_max_length is None:
        llm_model_max_length = 2048 # A conservative default if not found
        logger.warning(f"Could not determine LLM model_max_length, defaulting to {llm_model_max_length}.")
    else:
        logger.info(f"Determined LLM model_max_length: {llm_model_max_length}")
    
    # Update MIN_SEQ_POS, MAX_SEQ_POS based on actual LLM capabilities if desired,
    # but ensure normalization constants are consistent with training.
    # For this script, we'll assume training constants (MIN_SEQ_POS, MAX_SEQ_POS) are authoritative
    # for the length predictor's input normalization.
    # We use llm_model_max_length primarily for input truncation.


    attn_implementation = "eager"
    if USE_FLASH_ATTENTION_2:
        try:
            import flash_attn
            if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                attn_implementation = "flash_attention_2"
                logger.info("Flash Attention 2 will be used for LLM.")
            else:
                logger.info("Flash Attention 2 prerequisites not met, falling back to eager for LLM.")
        except ImportError:
            logger.info("Flash Attention 2 not installed, falling back to eager for LLM.")
    else:
        logger.info("USE_FLASH_ATTENTION_2 is False, using eager attention for LLM.")

    try:
        device_map = DEVICE_LLM if DEVICE_LLM.type == 'cuda' else None
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16 if DEVICE_LLM.type == 'cuda' and attn_implementation == "flash_attention_2" else torch.float32, # FA2 prefers bfloat16/float16
            device_map=device_map,
            attn_implementation=attn_implementation,
            output_hidden_states=True
        )
        if DEVICE_LLM.type == 'cpu' and attn_implementation == "flash_attention_2": # Should not happen if check above is correct
            llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float32, output_hidden_states=True).to(DEVICE_LLM)
        elif DEVICE_LLM.type == 'cuda' and not (DEVICE_LLM.index is None or DEVICE_LLM.index == llm_model.device.index) and device_map != "auto": # Check if model is on correct device after device_map
            # This check is tricky with device_map. For single GPU, device_map=DEVICE_LLM should handle it.
            pass
    except Exception as e:
        logger.error(f"Error loading LLM (attn: {attn_implementation}): {e}. Trying with eager attention and float32.")
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float32, # Fallback dtype
            device_map=DEVICE_LLM if DEVICE_LLM.type == 'cuda' else None,
            attn_implementation="eager",
            output_hidden_states=True
        ).to(DEVICE_LLM) # Ensure .to() if device_map is None (e.g. CPU)
    llm_model.eval()


    logger.info(f"Loading Length Predictor from: {LENGTH_PREDICTOR_PATH} on {DEVICE_PREDICTOR}")
    length_predictor = EnhancedMLP(HIDDEN_SIZE).to(DEVICE_PREDICTOR)
    try:
        # ... (model loading logic, same as your previous robust version, ensure to use logger.info/error) ...
        state_dict = torch.load(LENGTH_PREDICTOR_PATH, map_location=DEVICE_PREDICTOR)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_ddp_model = any(key.startswith('module.') for key in state_dict.keys())

        if is_ddp_model:
            logger.info("Detected DDP model state_dict for length predictor. Removing 'module.' prefix.")
            for k, v in state_dict.items(): new_state_dict[k[7:]] = v
            length_predictor.load_state_dict(new_state_dict)
        else:
            logger.info("Detected non-DDP model state_dict for length predictor.")
            length_predictor.load_state_dict(state_dict)
            
    except FileNotFoundError:
        logger.error(f"ERROR: Length predictor model file not found at {LENGTH_PREDICTOR_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading length predictor state_dict: {e}. Trying as checkpoint.")
        try:
            checkpoint = torch.load(LENGTH_PREDICTOR_PATH, map_location=DEVICE_PREDICTOR)
            state_dict = checkpoint['model_state_dict']
            # ... (rest of checkpoint loading logic with DDP prefix removal) ...
            new_state_dict = OrderedDict()
            is_ddp_model = any(key.startswith('module.') for key in state_dict.keys())
            if is_ddp_model:
                for k, v in state_dict.items(): new_state_dict[k[7:]] = v
                length_predictor.load_state_dict(new_state_dict)
            else:
                length_predictor.load_state_dict(state_dict)
            logger.info(f"Loaded length predictor from checkpoint (epoch {checkpoint.get('epoch', 'N/A')}).")
        except Exception as e_chk:
            logger.error(f"ERROR: Could not load length predictor: {e_chk}")
            return
    length_predictor.eval()


    prompts_data = load_prompts(PROMPT_SOURCE_FILE, NUM_TEST_PROMPTS)
    if not prompts_data:
        logger.error("No prompts to evaluate. Exiting.")
        return
        
    all_results_data = [] # Renamed to avoid conflict

    for prompt_info in tqdm(prompts_data, desc="Processing Prompts"):
        prompt_text = prompt_info["text"]
        prompt_id = prompt_info.get("id", f"p_{random.randint(1000,9999)}")

        for dec_params_idx, dec_params in enumerate(DECODING_PARAMS_LIST):
            current_max_new_tokens_session = dec_params['max_new_tokens'] # Max for this session
            
            # Tokenize prompt, ensuring space for generation within LLM's limits
            # Truncate from the left if prompt is too long
            inputs = llm_tokenizer(
                prompt_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=min(llm_model_max_length - 5, llm_model_max_length - current_max_new_tokens_session), # Ensure prompt+max_new fits
                padding=False # No padding needed for single sequence generation
            ).to(DEVICE_LLM)
            
            current_input_ids = inputs.input_ids # Will be appended with generated tokens
            original_prompt_len = current_input_ids.shape[1]
            
            if original_prompt_len == 0:
                logger.warning(f"Prompt '{prompt_id}' resulted in empty input_ids after tokenization/truncation, skipping.")
                continue

            # --- Store results for this specific (prompt, decoding_param_set) run ---
            session_generated_token_ids = []
            session_step_predictions = []
            session_eos_encountered = False
            actual_total_generated_steps = 0 # Actual steps taken before EOS or max_new_tokens

            # KV cache for LLM generation
            past_key_values = None

            # Note: This loop iterates for `current_max_new_tokens_session` steps *at most*.
            # It's simulating token-by-token generation.
            for step in range(current_max_new_tokens_session):
                current_full_sequence_len = original_prompt_len + step # Length *before* generating token for this step
                
                # Input for LLM: only the last token if past_key_values are used
                llm_step_input_ids = current_input_ids if step == 0 else current_input_ids[:, -1:]
                
                llm_outputs = llm_model(
                    input_ids=llm_step_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                last_token_embedding = llm_outputs.hidden_states[-1][:, -1, :].to(DEVICE_PREDICTOR) # [1, HIDDEN_SIZE]
                
                # Normalize parameters for length predictor input
                norm_params = normalize_eval_params(
                    dec_params['temperature'], dec_params['top_k'], dec_params['repetition_penalty'],
                    current_max_new_tokens_session, # This is the max_new_tokens for the *session*
                    current_full_sequence_len # current position in sequence (length of input to LLM)
                )

                predictor_input = {
                    'embedding': last_token_embedding.to(torch.float32),
                    'temperature': torch.tensor([norm_params['temperature']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    'top_k': torch.tensor([norm_params['top_k']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    'repetition_penalty': torch.tensor([norm_params['repetition_penalty']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    'max_len': torch.tensor([norm_params['max_len']], device=DEVICE_PREDICTOR, dtype=torch.float32), # Normalized max_len_session
                    'seq_pos': torch.tensor([norm_params['seq_pos']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                }
                
                start_time = time.perf_counter()
                if DEVICE_PREDICTOR.type == 'cuda': torch.cuda.synchronize(DEVICE_PREDICTOR)
                predicted_rest_len_tensor = length_predictor(predictor_input)
                if DEVICE_PREDICTOR.type == 'cuda': torch.cuda.synchronize(DEVICE_PREDICTOR)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                predicted_rest_len = predicted_rest_len_tensor.item()

                # LLM generates next token (for continuing the simulation)
                logits_for_next_token = llm_outputs.logits[:, -1, :] # Logits for the token *to be generated now*
                past_key_values = llm_outputs.past_key_values
                
                # Apply repetition penalty to logits_for_next_token if needed (as in your data gen)
                if dec_params['repetition_penalty'] != 1.0:
                    for token_id_in_generated_sequence in set(session_generated_token_ids): # Penalize already generated tokens
                        logits_for_next_token[0, token_id_in_generated_sequence] /= dec_params['repetition_penalty']

                next_token_id_tensor = sample_next_token_from_logits(
                    logits_for_next_token, 
                    dec_params['temperature'], 
                    dec_params['top_k']
                )
                next_token_id_item = next_token_id_tensor.item()
                session_generated_token_ids.append(next_token_id_item)
                actual_total_generated_steps = step + 1

                # Store step result (actual_rest_len will be post-calculated)
                session_step_predictions.append({
                    "step_index": step, # 0 to max_new_tokens-1
                    "current_full_sequence_len_for_pred": current_full_sequence_len,
                    "predicted_rest_len": predicted_rest_len,
                    "actual_rest_len": -1, # Placeholder, to be filled after full generation
                    "latency_ms": latency_ms,
                    "generated_token_id_at_this_step": next_token_id_item,
                })

                current_input_ids = torch.cat([current_input_ids, next_token_id_tensor], dim=-1)
                
                if next_token_id_item == llm_tokenizer.eos_token_id:
                    session_eos_encountered = True
                    logger.debug(f"EOS token encountered at step {step} for prompt {prompt_id}, params {dec_params_idx}")
                    break 
            
            # Post-process to calculate correct actual_rest_len for each step
            for step_pred_info in session_step_predictions:
                # actual_total_generated_steps is the number of tokens generated (1-indexed)
                # step_pred_info["step_index"] is the 0-indexed step number when prediction was made
                # The prediction at step_index `s` is for the tokens from `s+1` onwards.
                # Total tokens to be generated from this point = actual_total_generated_steps - (s+1)
                step_pred_info["actual_rest_len"] = max(0, actual_total_generated_steps - (step_pred_info["step_index"] + 1))
                step_pred_info["prediction_error"] = step_pred_info["predicted_rest_len"] - step_pred_info["actual_rest_len"]


            all_results_data.append({
                "prompt_id": prompt_id,
                "prompt_text_truncated": llm_tokenizer.decode(inputs.input_ids[0]), # Log the (potentially truncated) prompt fed to LLM
                "decoding_params_idx": dec_params_idx,
                "decoding_params": dec_params,
                "original_prompt_len_after_tokenization": original_prompt_len,
                "actual_generated_steps": actual_total_generated_steps,
                "eos_encountered_in_session": session_eos_encountered,
                "generated_token_ids_in_session": session_generated_token_ids,
                "step_predictions": session_step_predictions
            })
            
            del past_key_values
            if DEVICE_LLM.type == 'cuda': torch.cuda.empty_cache()


    # Save all detailed results
    with open(output_results_file, 'w') as f:
        json.dump(all_results_data, f, indent=2)
    logger.info(f"\nDetailed evaluation results saved to: {output_results_file}")

    # --- Aggregate Statistics ---
    logger.info("\n--- Aggregate Statistics ---")
    all_prediction_errors = []
    all_latencies = []
    for res_group in all_results_data:
        for step_res in res_group["step_predictions"]:
            all_prediction_errors.append(step_res["prediction_error"])
            all_latencies.append(step_res["latency_ms"])

    if all_prediction_errors:
        all_prediction_errors = np.array(all_prediction_errors)
        mae = np.mean(np.abs(all_prediction_errors))
        mse = np.mean(all_prediction_errors**2)
        rmse = np.sqrt(mse)
        bias = np.mean(all_prediction_errors)
        
        logger.info(f"Overall Prediction MAE: {mae:.2f} tokens")
        logger.info(f"Overall Prediction RMSE: {rmse:.2f} tokens")
        logger.info(f"Overall Prediction Bias (Pred - Actual): {bias:.2f} tokens")
    else:
        logger.info("No prediction errors recorded to aggregate.")

    if all_latencies:
        all_latencies = np.array(all_latencies)
        logger.info(f"\nPredictor Average Latency: {np.mean(all_latencies):.2f} ms")
        logger.info(f"Predictor Median Latency: {np.median(all_latencies):.2f} ms")
        logger.info(f"Predictor P90 Latency: {np.percentile(all_latencies, 90):.2f} ms")
        logger.info(f"Predictor P95 Latency: {np.percentile(all_latencies, 95):.2f} ms")
        logger.info(f"Predictor P99 Latency: {np.percentile(all_latencies, 99):.2f} ms")
    else:
        logger.info("No latencies recorded to aggregate.")
    logger.info("--------------------------")


if __name__ == "__main__":
    # --- Path Checks ---
    if "your_run_timestamp" in str(LENGTH_PREDICTOR_PATH) or \
    not LENGTH_PREDICTOR_PATH.exists():
        logger.error(f"LENGTH_PREDICTOR_PATH is a placeholder or does not exist: {LENGTH_PREDICTOR_PATH}")
        logger.error("Please update it to the actual path of your trained .pth file.")
        sys.exit(1) # Exit if critical path is not set
    
    # Optional: Check for prompt file if specified
    if PROMPT_SOURCE_FILE and not Path(PROMPT_SOURCE_FILE).exists():
        logger.warning(f"PROMPT_SOURCE_FILE specified but not found: {PROMPT_SOURCE_FILE}. Using default prompts.")
        PROMPT_SOURCE_FILE = None # Fallback to default

    evaluate_length_predictor()