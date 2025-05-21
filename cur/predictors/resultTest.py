import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import os
import random
import json
from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # For R2 score if needed
import sys # For exiting early
import logging # For better logging control
from datasets import load_dataset

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

HIDDEN_SIZE = 8192
MIN_TEMP, MAX_TEMP = 0.1, 0.9
MIN_TOP_K, MAX_TOP_K = 1, 100
MIN_REP_PENALTY, MAX_REP_PENALTY = 1.3, 1.6
MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS = 300, 500
MIN_SEQ_POS, MAX_SEQ_POS = 0, 8191

# configurations
LLM_MODEL_NAME = 'meta-llama/Meta-Llama-3-70B'
LENGTH_PREDICTOR_PATH = Path("./saved_models/20250509_003641/enhanced_mlp_best.pth") # Update this to your actual path
DS_CHOICE = 2
DS_NAME = 'yahma/alpaca-cleaned' # 0
DS_NAME_ALPACA_EVAL = 'tatsu-lab/alpaca_eval' # 1
DS_NAME_DOLLY = 'databricks/databricks-dolly-15k' # 2
USED_PROMPT_IDS_FILE = Path("./used_prompt_ids.txt")
# PROMPT_SOURCE_FILE = './test_prompts.json' # Path to a .json file with prompts, or use a default list
NUM_TEST_PROMPTS = 100
SAMPLE_PERCENTAGE = 0.01
ALPACA_CACHE_DIR = "./.cache/huggingface_datasets_eval_simple" # cache dir for datasets
EVAL_CACHE_DIR = "./.cache/huggingface_datasets_eval_extra"
OUTPUT_RESULTS_FILE_TEMPLATE = "./length_predictor_eval_results_{llm_name}_{timestamp}.jsonl" # Added timestamp
DEVICE_LLM = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_PREDICTOR = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_LLM)
USE_FLASH_ATTENTION_2 = False # Set to False to disable Flash Attention 2

# Decoding parameters to test
# IMPORTANT: Ensure these values are within the MIN_... and MAX_... ranges defined above for normalization.
DECODING_PARAMS_LIST = [
    {'temperature': 0.1, 'top_k': 1, 'repetition_penalty': 1.3, 'max_new_tokens': 300},
    {'temperature': 0.5, 'top_k': 100, 'repetition_penalty': 1.3, 'max_new_tokens': 300},
    {'temperature': 0.9, 'top_k': 10, 'repetition_penalty': 1.6, 'max_new_tokens': 500}, # Added another example
    {'temperature': 0.7, 'top_k': 50, 'repetition_penalty': 1.5, 'max_new_tokens': 400},
    {'temperature': 0.9, 'top_k': 10, 'repetition_penalty': 1.4, 'max_new_tokens': 300},
    {'temperature': 0.9, 'top_k': 50, 'repetition_penalty': 1.6, 'max_new_tokens': 300},
    {'temperature': 0.7, 'top_k': 10, 'repetition_penalty': 1.6, 'max_new_tokens': 300},
]

def load_used_prompt_ids(filepath: Path) -> set:
    """Loads a set of used prompt IDs from a file (one ID per line)."""
    used_ids = set()
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line: # non-empty line
                        try:
                            used_ids.add(int(line))
                        except ValueError:
                            logger.warning(f"Skipping invalid ID (not an int) in {filepath}: '{line}'")
            logger.info(f"Loaded {len(used_ids)} used prompt IDs from {filepath}")
        except Exception as e:
            logger.error(f"Error loading used prompt IDs from {filepath}: {e}")
    else:
        logger.warning(f"Used prompt IDs file not found: {filepath}. No prompts will be excluded based on this file.")
    return used_ids


def load_prompts_for_evaluation(
    ds_name: str, 
    sample_percentage: float, 
    num_prompts_limit: int, 
    cache_dir: str = None,
    used_ids_filepath: Path = USED_PROMPT_IDS_FILE
):
    logger.info(f"Loading dataset: {ds_name} for evaluation prompts...")
    
    ids_to_skip = load_used_prompt_ids(used_ids_filepath)
    try:
        temp_hf_dataset = load_dataset(ds_name, cache_dir=cache_dir)
        if 'train' not in temp_hf_dataset:
            logger.error(f"'train' split not found in {ds_name}. Available: {list(temp_hf_dataset.keys())}")
            return []
        dataset_split = temp_hf_dataset['train'] 
        logger.info(f"Using 'train' split from {ds_name}.")
    except Exception as e:
        logger.error(f"Error loading dataset {ds_name}: {e}")
        return []
    prompt_template_with_input = "{instruction}\n\n{input}"
    prompt_template_no_input = "{instruction}"
    
    all_formatted_prompts = [] 
    skipped_due_to_used_id = 0
    logger.info("Formatting prompts from the dataset and filtering used IDs...")
    
    # enumerate(dataset_split): (index, entry)
    for original_idx, entry in enumerate(dataset_split):
        if original_idx in ids_to_skip:
            skipped_due_to_used_id += 1
            # logger.debug(f"Skipping prompt with original_idx {original_idx} as it was used for training.")
            continue 
        
        instruction = entry.get("instruction", "")
        inp_text = entry.get("input", "")
        
        if not instruction:
            # logger.debug(f"Skipping entry with original_idx {original_idx} due to empty instruction.")
            continue
        
        if inp_text and inp_text.strip():
            prompt_text = prompt_template_with_input.format(instruction=instruction, input=inp_text)
        else:
            prompt_text = prompt_template_no_input.format(instruction=instruction)
        
        safe_ds_name = ds_name.split('/')[-1].replace('-', '_')
        prompt_id = f"{safe_ds_name}_train_{original_idx}" 
        all_formatted_prompts.append({"id": prompt_id, "text": prompt_text, "original_idx": original_idx}) # 可选：保存原始索引
    if skipped_due_to_used_id > 0:
        logger.info(f"Skipped {skipped_due_to_used_id} prompts because their original_idx was in the used IDs list.")
    if not all_formatted_prompts:
        logger.warning(f"No prompts remaining after filtering from dataset {ds_name}.")
        return []
    total_prompts_before_sampling = len(all_formatted_prompts)
    logger.info(f"Total formatted and available prompts (after filtering used IDs): {total_prompts_before_sampling}")
    
    # Sample prompts based on the sample_percentage
    sampled_prompts = []
    if not (0 <= sample_percentage <= 1.0):
        logger.warning(f"Sample percentage {sample_percentage*100:.2f}% is out of [0, 100] range. Using all available prompts or limiting by num_prompts_limit.")
        sampled_prompts = all_formatted_prompts
    elif sample_percentage == 0:
        logger.info("Sample percentage is 0. No prompts will be selected by sampling.")
    else:
        num_to_sample = int(round(total_prompts_before_sampling * sample_percentage))
        if num_to_sample == 0 and total_prompts_before_sampling > 0: num_to_sample = 1
        
        if num_to_sample >= total_prompts_before_sampling:
            sampled_prompts = all_formatted_prompts
        else:
            # random.seed(42) # Uncomment for reproducible sampling
            sampled_prompts = random.sample(all_formatted_prompts, num_to_sample)
        logger.info(f"Sampled {len(sampled_prompts)} prompts ({sample_percentage*100:.2f}% of {total_prompts_before_sampling}).")
    final_prompts_for_eval = sampled_prompts
    if num_prompts_limit is not None and len(sampled_prompts) > num_prompts_limit:
        final_prompts_for_eval = sampled_prompts[:num_prompts_limit] 
        logger.info(f"Limited prompts from {len(sampled_prompts)} to {len(final_prompts_for_eval)} by num_prompts_limit ({num_prompts_limit}).")
    
    if final_prompts_for_eval:
        logger.info(f"Final number of prompts selected for evaluation: {len(final_prompts_for_eval)}")
    else:
        logger.warning("No prompts selected for evaluation after sampling, filtering, and limiting.")
    return final_prompts_for_eval

def load_alpaca_eval_prompts(
    sample_percentage: float, 
    num_prompts_limit: int, 
    cache_dir: str = None,
    used_ids_set: set = None
):
    """
    Loads prompts from tatsu-lab/alpaca_eval ('eval' split).
    The 'instruction' field is used as the prompt text.
    'dataset' field can be used to create a more specific ID if needed.
    """
    ds_name = DS_NAME_ALPACA_EVAL
    logger.info(f"Loading dataset: {ds_name} for evaluation prompts...")
    if used_ids_set is None:
        used_ids_set = set()
        
    try:
        temp_hf_dataset = load_dataset(ds_name, "alpaca_eval", cache_dir=cache_dir) # 指定子集名称
        if 'eval' not in temp_hf_dataset:
            logger.error(f"'eval' split not found in {ds_name} (subset 'alpaca_eval'). Available: {list(temp_hf_dataset.keys())}")
            return []
        dataset_split = temp_hf_dataset['eval']
        logger.info(f"Using 'eval' split from {ds_name}.")
    except Exception as e:
        logger.error(f"Error loading dataset {ds_name}: {e}")
        return []
    
    all_formatted_prompts = []
    skipped_due_to_used_id = 0
    logger.info("Formatting prompts from alpaca_eval dataset and filtering used IDs...")
    
    for original_idx, entry in enumerate(dataset_split):
        if original_idx in used_ids_set: # assume used_ids_set is a set of original_idx
            skipped_due_to_used_id += 1
            continue
        
        instruction = entry.get("instruction", "")
        if not instruction:
            logger.debug(f"Skipping entry with original_idx {original_idx} from alpaca_eval due to empty instruction.")
            continue
        
        prompt_text = instruction # directly use instruction as prompt text
        
        # ID could be based on dataset name and original index
        source_dataset_field = entry.get("dataset", "unknown_source")
        prompt_id = f"alpaca_eval_{source_dataset_field}_{original_idx}"
        all_formatted_prompts.append({"id": prompt_id, "text": prompt_text, "original_idx": original_idx})
    
    if skipped_due_to_used_id > 0:
        logger.info(f"Skipped {skipped_due_to_used_id} prompts from alpaca_eval (already used).")
    if not all_formatted_prompts:
        logger.warning("No prompts remaining after filtering from alpaca_eval.")
        return []
    
    total_prompts_before_sampling = len(all_formatted_prompts)
    logger.info(f"Total formatted and available prompts from alpaca_eval (after filtering): {total_prompts_before_sampling}")
    
    sampled_prompts = []
    if not (0 <= sample_percentage <= 1.0):
        logger.warning(f"alpaca_eval: Sample percentage {sample_percentage*100:.2f}% out of range. Using all or limit.")
        sampled_prompts = all_formatted_prompts
    elif sample_percentage == 0: logger.info("alpaca_eval: Sample percentage is 0.")
    else:
        num_to_sample = int(round(total_prompts_before_sampling * sample_percentage))
        if num_to_sample == 0 and total_prompts_before_sampling > 0: num_to_sample = 1
        if num_to_sample >= total_prompts_before_sampling: sampled_prompts = all_formatted_prompts
        else: sampled_prompts = random.sample(all_formatted_prompts, num_to_sample)
        logger.info(f"alpaca_eval: Sampled {len(sampled_prompts)} prompts.")
    
    final_prompts_for_eval = sampled_prompts
    if num_prompts_limit is not None and len(sampled_prompts) > num_prompts_limit:
        final_prompts_for_eval = sampled_prompts[:num_prompts_limit]
        logger.info(f"alpaca_eval: Limited prompts to {len(final_prompts_for_eval)} by num_prompts_limit.")
    
    if final_prompts_for_eval: logger.info(f"alpaca_eval: Final prompts for evaluation: {len(final_prompts_for_eval)}")
    else: logger.warning("alpaca_eval: No prompts selected after sampling/limiting.")
    return final_prompts_for_eval


def load_dolly_v2_prompts(
    sample_percentage: float, 
    num_prompts_limit: int, 
    cache_dir: str = None,
    used_ids_set: set = None # 可选
):
    """
    Loads prompts from databricks/dolly-v2-12k ('train' split).
    Combines 'instruction' and 'context' (if present) to form the prompt.
    'category' can be used for ID or filtering.
    """
    ds_name = DS_NAME_DOLLY
    logger.info(f"Loading dataset: {ds_name} for evaluation prompts...")
    if used_ids_set is None:
        used_ids_set = set()
    
    try:
        temp_hf_dataset = load_dataset(ds_name, cache_dir=cache_dir)
        if 'train' not in temp_hf_dataset:
            logger.error(f"'train' split not found in {ds_name}. Available: {list(temp_hf_dataset.keys())}")
            return []
        dataset_split = temp_hf_dataset['train']
        logger.info(f"Using 'train' split from {ds_name}.")
    except Exception as e:
        logger.error(f"Error loading dataset {ds_name}: {e}")
        return []
    
    all_formatted_prompts = []
    skipped_due_to_used_id = 0
    logger.info("Formatting prompts from dolly-v2-12k dataset and filtering used IDs...")
    for original_idx, entry in enumerate(dataset_split):
        if original_idx in used_ids_set:
            skipped_due_to_used_id += 1
            continue
        instruction = entry.get("instruction", "")
        context = entry.get("context", "") # Dolly has a 'context' field
        category = entry.get("category", "unknown_category")
        if not instruction:
            logger.debug(f"Skipping entry with original_idx {original_idx} from dolly due to empty instruction.")
            continue
        
        if context and context.strip():
            # Example: combine context and instruction
            prompt_text = f"Context: {context}\n\nInstruction: {instruction}"
            # Or, you might choose to only use instruction if context is too long, or based on category
        else:
            prompt_text = instruction
        
        prompt_id = f"dolly_v2_{category}_{original_idx}"
        all_formatted_prompts.append({"id": prompt_id, "text": prompt_text, "original_idx": original_idx})
    if skipped_due_to_used_id > 0:
        logger.info(f"Skipped {skipped_due_to_used_id} prompts from dolly-v2 (already used).")
    if not all_formatted_prompts:
        logger.warning("No prompts remaining after filtering from dolly-v2.")
        return []
    total_prompts_before_sampling = len(all_formatted_prompts)
    logger.info(f"Total formatted and available prompts from dolly-v2 (after filtering): {total_prompts_before_sampling}")
    sampled_prompts = [] # Placeholder for actual sampling logic
    if not (0 <= sample_percentage <= 1.0):
        logger.warning(f"dolly: Sample percentage {sample_percentage*100:.2f}% out of range. Using all or limit.")
        sampled_prompts = all_formatted_prompts
    elif sample_percentage == 0: logger.info("dolly: Sample percentage is 0.")
    else:
        num_to_sample = int(round(total_prompts_before_sampling * sample_percentage))
        if num_to_sample == 0 and total_prompts_before_sampling > 0: num_to_sample = 1
        if num_to_sample >= total_prompts_before_sampling: sampled_prompts = all_formatted_prompts
        else: sampled_prompts = random.sample(all_formatted_prompts, num_to_sample)
        logger.info(f"dolly: Sampled {len(sampled_prompts)} prompts.")
    final_prompts_for_eval = sampled_prompts
    if num_prompts_limit is not None and len(sampled_prompts) > num_prompts_limit:
        final_prompts_for_eval = sampled_prompts[:num_prompts_limit]
        logger.info(f"dolly: Limited prompts to {len(final_prompts_for_eval)} by num_prompts_limit.")
    
    if final_prompts_for_eval: logger.info(f"dolly: Final prompts for evaluation: {len(final_prompts_for_eval)}")
    else: logger.warning("dolly: No prompts selected after sampling/limiting.")
    return final_prompts_for_eval

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
    
    print(f"Normalizing parameters: temp={temp}, top_k={top_k_val}, rep_p={rep_p}, max_len_session={max_len_session}, current_seq_pos_val={current_seq_pos_val}")
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
    # --- Setup Run Information and Output File ---
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Ensure template uses .jsonl if writing line by line
    output_results_filename = OUTPUT_RESULTS_FILE_TEMPLATE.format(
        llm_name=LLM_MODEL_NAME.split('/')[-1].replace('-', '_'), # Make filename safe
        timestamp=run_timestamp
    )
    output_results_file = Path(output_results_filename)
    output_results_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluation Run ID: {run_timestamp}")
    logger.info(f"LLM for generation: {LLM_MODEL_NAME} on {DEVICE_LLM}")
    logger.info(f"Length predictor model path: {LENGTH_PREDICTOR_PATH} on {DEVICE_PREDICTOR}")
    logger.info(f"Attempting to use Flash Attention 2 for LLM: {USE_FLASH_ATTENTION_2}")
    logger.info(f"Outputting detailed results (JSONL) to: {output_results_file}")
    
    # --- 1. Load LLM and Tokenizer ---
    logger.info(f"Loading tokenizer for: {LLM_MODEL_NAME}")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        logger.critical(f"Failed to load LLM tokenizer for {LLM_MODEL_NAME}: {e}")
        return
    
    if llm_tokenizer.pad_token_id is None:
        if llm_tokenizer.eos_token_id is not None:
            llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
            logger.info(f"Set LLM pad_token_id to eos_token_id: {llm_tokenizer.eos_token_id}")
        else:
            # Add a new pad token if no eos token either (less common for decoders)
            llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info(f"Added a new pad token '[PAD]' to LLM tokenizer.")
    
    # --- Robustly determine llm_model_max_length ---
    llm_model_max_len = None
    try:
        llm_config_obj = AutoConfig.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        llm_model_max_len = getattr(llm_config_obj, 'max_position_embeddings', None)
    except Exception as e_conf:
        logger.warning(f"Could not load AutoConfig for {LLM_MODEL_NAME} to get max_position_embeddings ({e_conf}).")

    if llm_model_max_len is None:
        llm_model_max_len = getattr(llm_tokenizer, 'model_max_length', None)
    if llm_model_max_len is None:
        llm_model_max_len = llm_tokenizer.tokenizer_config.get('model_max_length', None)
        
    SAFE_LLM_MAX_LENGTH = 8192 # Default for Llama 3 like models
    # KNOWN_MODEL_MAX_LENGTHS could be a dict for specific model overrides if needed.
    
    if not isinstance(llm_model_max_len, int) or llm_model_max_len <= 0:
        logger.warning(f"Could not reliably determine model_max_length from tokenizer/config (found: {llm_model_max_len}).")
        llm_model_max_len = KNOWN_MODEL_MAX_LENGTHS.get(LLM_MODEL_NAME, SAFE_LLM_MAX_LENGTH) if 'KNOWN_MODEL_MAX_LENGTHS' in globals() else SAFE_LLM_MAX_LENGTH
        logger.warning(f"Setting LLM max_length to: {llm_model_max_len}")
    elif llm_model_max_len > 32768: # Arbitrary upper cap for sanity
        logger.warning(f"Determined model_max_length {llm_model_max_len} seems excessively large. Capping at 32768 for safety during tokenization.")
        llm_model_max_len = 32768
        
    logger.info(f"Using effective LLM model_max_length for input truncation: {llm_model_max_len}")
    # Note: The global constant MAX_SEQ_POS (e.g., 8191) is used for normalizing the 'seq_pos' feature
    # for the length predictor and should be consistent with its training.
    # llm_model_max_len is for LLM's own input sequence length handling.
    logger.info(f"Note: Global MAX_SEQ_POS for length predictor normalization is {MAX_SEQ_POS}")
    
    # --- Configure Attention Implementation for LLM ---
    attn_implementation = "eager" # Default
    if USE_FLASH_ATTENTION_2 and torch.cuda.is_available():
        try:
            import flash_attn 
            # Check if PyTorch version supports SDPA (scaled_dot_product_attention)
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                attn_implementation = "flash_attention_2"
                logger.info("Attempting to use Flash Attention 2 for LLM.")
            else:
                logger.info("Current PyTorch version does not support scaled_dot_product_attention needed for Flash Attention 2.")
        except ImportError:
            logger.info("flash_attn library not installed. Cannot use 'flash_attention_2'.")
    else:
        logger.info(f"Not using Flash Attention 2 (USE_FLASH_ATTENTION_2: {USE_FLASH_ATTENTION_2}, CUDA available: {torch.cuda.is_available()}). Using 'eager' attention for LLM.")
    
    # --- Load LLM ---
    try:
        # llm_device_map_arg = DEVICE_LLM if DEVICE_LLM.type == 'cuda' else None
        llm_device_map_arg = "auto"
        # For very large models on a single GPU, quantization is often necessary.
        # If `LOAD_IN_4BIT` (from your previous iteration) is True, add quantization_config here.
        # For this version, I'm assuming no explicit quantization config for simplicity based on current code.
        
        llm_torch_dtype = torch.float32 # Default
        if DEVICE_LLM.type == 'cuda':
            llm_torch_dtype = torch.bfloat16 if attn_implementation == "flash_attention_2" else torch.float16
            # If not using FA2, float16 might be a good compromise for speed/memory on CUDA for 70B if bf16 not fully stable or desired.
            # Llama models often default to bfloat16 or float16 on CUDA.
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=llm_torch_dtype,
            device_map=llm_device_map_arg, # Handles multi-GPU if "auto" or sharding if single GPU memory is tight (with accelerate)
            attn_implementation=attn_implementation,
            output_hidden_states=True, # Required for embeddings
            trust_remote_code=True 
        )
        if llm_device_map_arg is None and DEVICE_LLM.type != 'cpu': # e.g., for MPS or if device_map was None for CPU
            llm_model = llm_model.to(DEVICE_LLM)
        logger.info(f"LLM loaded on its device(s) (model.device: {llm_model.device}) with {attn_implementation} attention.")
    except Exception as e_load_llm:
        logger.error(f"Error loading LLM (attn: {attn_implementation}, device_map: {llm_device_map_arg}): {e_load_llm}.")
        logger.error("Attempting fallback: eager attention, float32, and explicit .to(DEVICE_LLM).")
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                output_hidden_states=True,
                trust_remote_code=True
            ).to(DEVICE_LLM)
            logger.info(f"LLM loaded (fallback) successfully on {llm_model.device} with eager attention.")
        except Exception as e_fallback_llm:
            logger.critical(f"CRITICAL: Fallback LLM loading also failed: {e_fallback_llm}")
            return # Cannot proceed without LLM
    llm_model.eval()
    
    # --- 2. Load Length Predictor Model ---
    logger.info(f"Loading Length Predictor from: {LENGTH_PREDICTOR_PATH} on {DEVICE_PREDICTOR}")
    length_predictor = EnhancedMLP(HIDDEN_SIZE).to(DEVICE_PREDICTOR)
    try:
        state_dict = torch.load(LENGTH_PREDICTOR_PATH, map_location=DEVICE_PREDICTOR)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        # Check if the state_dict is from a DDP model (keys start with 'module.')
        is_ddp_model = any(key.startswith('module.') for key in state_dict.keys())
        if is_ddp_model:
            logger.debug("Detected DDP model state_dict for length predictor. Removing 'module.' prefix.")
            for k, v in state_dict.items(): new_state_dict[k[7:]] = v # Remove 'module.'
            length_predictor.load_state_dict(new_state_dict)
        else:
            length_predictor.load_state_dict(state_dict)
        logger.info("Length predictor loaded successfully.")
    except FileNotFoundError:
        logger.critical(f"CRITICAL: Length predictor model file not found at {LENGTH_PREDICTOR_PATH}")
        return
    except Exception as e_load_pred: 
        logger.error(f"Error loading length predictor state_dict: {e_load_pred}. Trying to load as checkpoint.")
        try:
            checkpoint = torch.load(LENGTH_PREDICTOR_PATH, map_location=DEVICE_PREDICTOR)
            state_dict = checkpoint['model_state_dict']
            new_state_dict = OrderedDict()
            is_ddp_model_chk = any(key.startswith('module.') for key in state_dict.keys())
            if is_ddp_model_chk:
                for k_chk, v_chk in state_dict.items(): new_state_dict[k_chk[7:]] = v_chk
                length_predictor.load_state_dict(new_state_dict)
            else:
                length_predictor.load_state_dict(state_dict)
            logger.info(f"Length predictor loaded from checkpoint (epoch {checkpoint.get('epoch', 'N/A')}).")
        except Exception as e_chk_load:
            logger.critical(f"CRITICAL: Could not load length predictor from file or checkpoint: {e_chk_load}")
            return
    length_predictor.eval() # Set predictor to evaluation mode
    
    # --- 3. Load Prompts ---
    prompts_data = []
    
    if DS_CHOICE == 0: # "alpaca_clean"
        prompts_data = load_prompts_for_evaluation(
            ds_name=DS_NAME,
            sample_percentage=SAMPLE_PERCENTAGE,
            num_prompts_limit=NUM_TEST_PROMPTS,
            cache_dir=ALPACA_CACHE_DIR,
            used_ids_filepath=USED_PROMPT_IDS_FILE
        )
    elif DS_CHOICE == 1: # "alpaca_eval"
        prompts_data = load_alpaca_eval_prompts(
            sample_percentage=SAMPLE_PERCENTAGE,
            num_prompts_limit=NUM_TEST_PROMPTS,
            cache_dir=EVAL_CACHE_DIR,
            used_ids_set=None
        )
    elif DS_CHOICE == 2: # "dolly_v2"
        prompts_data = load_dolly_v2_prompts(
            sample_percentage=SAMPLE_PERCENTAGE,
            num_prompts_limit=NUM_TEST_PROMPTS,
            cache_dir=EVAL_CACHE_DIR,
            used_ids_set=None
        )
    else:
        logger.error(f"Invalid dataset choice: {DS_CHOICE}. Must be 0 (alpaca_clean), 1 (alpaca_eval), or 2 (dolly_v2).")
    
    if not prompts_data:
        logger.critical("Failed to load any prompts for evaluation. Exiting.")
        sys.exit(1)
    
    logger.info(f"Starting evaluation with {len(prompts_data)} prompts and {len(DECODING_PARAMS_LIST)} decoding parameter sets.")
    
    # --- 4. Evaluation Loop - Results Written Line-by-Line ---
    # Open file once to write all results as JSON Lines
    with open(output_results_file, 'w') as f_out:
        for prompt_idx, prompt_info in enumerate(tqdm(prompts_data, desc="Processing Prompts")):
            prompt_text = prompt_info["text"]
            prompt_id = prompt_info["id"]
            
            for dec_params_idx, dec_params in enumerate(DECODING_PARAMS_LIST):
                current_max_new_tokens_session = dec_params['max_new_tokens']
                
                # Validate if current_max_new_tokens_session is within normalization range for 'max_len_session'
                if not (MIN_MAX_NEW_TOKENS <= current_max_new_tokens_session <= MAX_MAX_NEW_TOKENS):
                    logger.warning(
                        f"For prompt {prompt_id}, params_idx {dec_params_idx}: "
                        f"Session max_new_tokens {current_max_new_tokens_session} is outside "
                        f"defined normalization range [{MIN_MAX_NEW_TOKENS}, {MAX_MAX_NEW_TOKENS}]. "
                        "Length predictor's 'max_len' feature will be clipped during normalization, affecting prediction."
                    )
                
                # Calculate max length for tokenizer, ensuring space for generated tokens
                # Subtract a small buffer (e.g., 5-10 tokens) for special tokens or edge cases.
                buffer_tokens = 5 
                max_prompt_len_for_tokenizer = llm_model_max_len - current_max_new_tokens_session - buffer_tokens
                
                if max_prompt_len_for_tokenizer <= 0:
                    logger.warning(
                        f"Prompt {prompt_id} with max_new_tokens {current_max_new_tokens_session} (LLM max: {llm_model_max_len}) "
                        f"results in non-positive max_prompt_len_for_tokenizer ({max_prompt_len_for_tokenizer}). "
                        "Skipping this parameter set for this prompt."
                    )
                    continue # Skip this (prompt, dec_param) combination
                
                # Tokenize the prompt
                inputs = llm_tokenizer(
                    prompt_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=max_prompt_len_for_tokenizer, 
                    padding=False # Not needed for single sequence generation
                ).to(DEVICE_LLM)
                
                current_input_ids = inputs.input_ids
                original_prompt_len_tokenized = current_input_ids.shape[1]
                
                if original_prompt_len_tokenized == 0:
                    logger.warning(f"Prompt '{prompt_id}' (params_idx {dec_params_idx}) resulted in empty input_ids after tokenization/truncation. Skipping.")
                    continue
                
                # --- Store results for this specific (prompt, decoding_param_set) run ---
                session_generated_token_ids = []
                session_step_predictions = []
                session_eos_encountered = False
                actual_total_generated_steps = 0 # How many tokens were actually generated
                past_key_values = None # KV cache
                
                # --- Decoding Loop (token by token) ---
                for step in range(current_max_new_tokens_session):
                    current_full_sequence_len = original_prompt_len_tokenized + step # Length *before* generating token for this step
                    
                    # Safety break if sequence approaches LLM's absolute max length
                    if current_full_sequence_len >= llm_model_max_len -1: # Leave one spot for next token
                        logger.warning(
                            f"Sequence length {current_full_sequence_len} approaching LLM max {llm_model_max_len} "
                            f"for prompt {prompt_id}. Stopping generation for this (prompt, params) early."
                        )
                        break
                    
                    # Prepare input for LLM (only the last token if past_key_values are available)
                    llm_step_input_ids = current_input_ids if step == 0 else current_input_ids[:, -1:]
                    
                    try:
                        llm_outputs = llm_model(
                            input_ids=llm_step_input_ids, 
                            past_key_values=past_key_values,
                            use_cache=True, 
                            output_hidden_states=True, 
                            return_dict=True
                        )
                    except Exception as e_llm_fwd:
                        logger.error(f"Error during LLM forward pass for prompt {prompt_id}, step {step}: {e_llm_fwd}")
                        session_eos_encountered = True # Mark as ended to stop further processing for this combo
                        break # Stop generation for this (prompt, params) combination
                    
                    # Get embedding for the length predictor (from the last token of the input to LLM)
                    last_token_embedding = llm_outputs.hidden_states[-1][:, -1, :].to(DEVICE_PREDICTOR) # Shape: [1, HIDDEN_SIZE]
                    
                    # Normalize parameters for the length predictor
                    norm_params = normalize_eval_params(
                        dec_params['temperature'], dec_params['top_k'], dec_params['repetition_penalty'],
                        current_max_new_tokens_session, # The 'max_len' context for the predictor
                        current_full_sequence_len     # The current position
                    )
                    
                    predictor_input = {
                        'embedding': last_token_embedding.to(torch.float32), # Ensure float32
                        'temperature': torch.tensor([norm_params['temperature']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                        'top_k': torch.tensor([norm_params['top_k']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                        'repetition_penalty': torch.tensor([norm_params['repetition_penalty']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                        'max_len': torch.tensor([norm_params['max_len']], device=DEVICE_PREDICTOR, dtype=torch.float32), # Normalized session max_len
                        'seq_pos': torch.tensor([norm_params['seq_pos']], device=DEVICE_PREDICTOR, dtype=torch.float32),
                    }
                    
                    # --- Predict remaining length and measure latency ---
                    pred_start_time = time.perf_counter()
                    if DEVICE_PREDICTOR.type == 'cuda': torch.cuda.synchronize(DEVICE_PREDICTOR)
                    predicted_rest_len_tensor = length_predictor(predictor_input)
                    if DEVICE_PREDICTOR.type == 'cuda': torch.cuda.synchronize(DEVICE_PREDICTOR)
                    pred_end_time = time.perf_counter()
                    
                    latency_ms = (pred_end_time - pred_start_time) * 1000
                    predicted_rest_len = predicted_rest_len_tensor.item() # Scalar value
                    
                    # --- LLM generates the next token for the ongoing sequence ---
                    logits_for_next_token = llm_outputs.logits[:, -1, :] # Logits for the token to be generated now
                    past_key_values = llm_outputs.past_key_values # Update KV cache
                    
                    # Apply repetition penalty manually to logits (consistent with your data generation)
                    if dec_params['repetition_penalty'] != 1.0 and session_generated_token_ids:
                        # Convert to tensor only if not empty for efficient indexing
                        unique_generated_ids_tensor = torch.tensor(list(set(session_generated_token_ids)), 
                                                                    device=logits_for_next_token.device, dtype=torch.long)
                        if unique_generated_ids_tensor.numel() > 0:
                            logits_for_next_token[0, unique_generated_ids_tensor] /= dec_params['repetition_penalty']
                    
                    # Sample the next token from LLM's logits
                    next_token_id_tensor = sample_next_token_from_logits(
                        logits_for_next_token, 
                        dec_params['temperature'], 
                        dec_params['top_k']
                    )
                    next_token_id_item = next_token_id_tensor.item()
                    
                    session_generated_token_ids.append(next_token_id_item)
                    actual_total_generated_steps = step + 1 # Update how many tokens have been generated so far
                    
                    # Store results for this step (actual_rest_len will be post-calculated)
                    session_step_predictions.append({
                        "step_index": step, # 0-indexed step number
                        "current_full_sequence_len_for_pred": current_full_sequence_len,
                        "predicted_rest_len": predicted_rest_len,
                        "actual_rest_len": -1, # Placeholder, will be filled after full generation for this (prompt,params)
                        "latency_ms": latency_ms,
                        "generated_token_id_at_this_step": next_token_id_item,
                    })
                    
                    # Append the newly generated token to current_input_ids for the next LLM step
                    current_input_ids = torch.cat([current_input_ids, next_token_id_tensor.to(current_input_ids.device)], dim=-1)
                    
                    # Check for EOS token
                    if next_token_id_item == llm_tokenizer.eos_token_id:
                        session_eos_encountered = True
                        logger.debug(f"EOS token encountered at step {step} for prompt {prompt_id}, params_idx {dec_params_idx}")
                        break # Break from the inner token generation loop
                
                # --- Post-process to calculate correct actual_rest_len for each step in this session ---
                for step_pred_info in session_step_predictions:
                    # actual_total_generated_steps is the number of tokens generated (1-indexed)
                    # step_pred_info["step_index"] is the 0-indexed step number when prediction was made
                    # The prediction at step_index `s` is for the tokens from `s+1` onwards.
                    # True remaining tokens from this point = actual_total_generated_steps - (s+1)
                    step_pred_info["actual_rest_len"] = max(0, actual_total_generated_steps - (step_pred_info["step_index"] + 1))
                    step_pred_info["prediction_error"] = step_pred_info["predicted_rest_len"] - step_pred_info["actual_rest_len"]
                
                # --- Prepare the result dictionary for this (prompt, dec_params) combination ---
                current_session_result_dict = {
                    "prompt_id": prompt_id,
                    "prompt_tokenized_len": original_prompt_len_tokenized, # Length of the (potentially truncated) prompt fed to LLM
                    "decoding_params_idx": dec_params_idx,
                    "decoding_params": dec_params,
                    "actual_generated_steps": actual_total_generated_steps, # How many tokens were actually generated
                    "eos_encountered_in_session": session_eos_encountered,
                    # "generated_token_ids_in_session": session_generated_token_ids, # Can be very long, uncomment if needed for debug
                    "step_predictions": session_step_predictions # List of per-step prediction details
                }
                
                # --- Write this session's result as a JSON Line ---
                f_out.write(json.dumps(current_session_result_dict) + '\n')
                
                # Optional: Flush frequently for long runs to see intermediate results, but impacts performance.
                # if (dec_params_idx + 1) % 5 == 0: f_out.flush() 
                
                # --- Clean up GPU memory for the next iteration ---
                del past_key_values, llm_outputs, last_token_embedding, predictor_input, predicted_rest_len_tensor, logits_for_next_token
                if DEVICE_LLM.type == 'cuda':
                    try:
                        # Set current device to DEVICE_LLM before emptying cache
                        with torch.cuda.device(DEVICE_LLM):
                            torch.cuda.empty_cache()
                        # logger.debug(f"Cache emptied for DEVICE_LLM: {DEVICE_LLM}")
                    except Exception as e_cache_llm:
                        logger.warning(f"Could not empty cache for DEVICE_LLM ({DEVICE_LLM}): {e_cache_llm}")

                if DEVICE_PREDICTOR.type == 'cuda' and DEVICE_PREDICTOR != DEVICE_LLM:
                    try:
                        # Set current device to DEVICE_PREDICTOR before emptying cache
                        with torch.cuda.device(DEVICE_PREDICTOR):
                            torch.cuda.empty_cache()
                        # logger.debug(f"Cache emptied for DEVICE_PREDICTOR: {DEVICE_PREDICTOR}")
                    except Exception as e_cache_pred:
                        logger.warning(f"Could not empty cache for DEVICE_PREDICTOR ({DEVICE_PREDICTOR}): {e_cache_pred}")
            # End of dec_params loop
        # End of prompts_data loop
    # End of file writing context
    
    logger.info(f"\nDetailed evaluation results saved line-by-line to: {output_results_file}")
    
    # --- 5. Aggregate Statistics from the saved JSONL file ---
    logger.info(f"\n--- Aggregating Statistics from {output_results_file} ---")
    all_prediction_errors_agg = []
    all_latencies_agg = []
    
    if output_results_file.exists() and output_results_file.stat().st_size > 0:
        with open(output_results_file, 'r') as f_in:
            for line_idx, line in enumerate(f_in):
                try:
                    res_group = json.loads(line) # Each line is a JSON object for one (prompt, dec_params) run
                    if "step_predictions" in res_group and isinstance(res_group["step_predictions"], list):
                        for step_res in res_group["step_predictions"]:
                            if "prediction_error" in step_res:
                                all_prediction_errors_agg.append(step_res["prediction_error"])
                            if "latency_ms" in step_res:
                                all_latencies_agg.append(step_res["latency_ms"])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {line_idx+1} in {output_results_file}")
                    continue
    else:
        logger.warning(f"Results file {output_results_file} is empty or does not exist. Cannot aggregate statistics.")
    
    # --- Print Aggregate Statistics ---
    if all_prediction_errors_agg:
        all_prediction_errors_np = np.array(all_prediction_errors_agg)
        mae = np.mean(np.abs(all_prediction_errors_np))
        mse = np.mean(all_prediction_errors_np**2)
        rmse = np.sqrt(mse)
        bias = np.mean(all_prediction_errors_np) # Average error (Pred - Actual)
        logger.info(f"Overall Prediction MAE (from file): {mae:.2f} tokens")
        logger.info(f"Overall Prediction RMSE (from file): {rmse:.2f} tokens")
        logger.info(f"Overall Prediction Bias (Pred - Actual) (from file): {bias:.2f} tokens")
        logger.info(f"Number of prediction data points analyzed: {len(all_prediction_errors_np)}")
    else:
        logger.info("No valid prediction errors found in results file to aggregate.")
    
    if all_latencies_agg:
        all_latencies_np = np.array(all_latencies_agg)
        logger.info(f"Predictor Average Latency (from file): {np.mean(all_latencies_np):.2f} ms")
        logger.info(f"Predictor Median Latency (from file): {np.median(all_latencies_np):.2f} ms")
        logger.info(f"Predictor P90 Latency (from file): {np.percentile(all_latencies_np, 90):.2f} ms")
        logger.info(f"Predictor P95 Latency (from file): {np.percentile(all_latencies_np, 95):.2f} ms")
        logger.info(f"Predictor P99 Latency (from file): {np.percentile(all_latencies_np, 99):.2f} ms")
        logger.info(f"Number of latency data points analyzed: {len(all_latencies_np)}")
    else:
        logger.info("No valid latencies found in results file to aggregate.")
    logger.info("--- Evaluation Complete ---")


if __name__ == "__main__":
    # --- Environment Setup for PyTorch (Optional, if facing fragmentation OOM) ---
    # prev_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # logger.info(f"Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (was: {prev_alloc_conf})")
    
    # --- Path Checks ---
    if not LENGTH_PREDICTOR_PATH.exists() or \
        ("your_run_timestamp" in str(LENGTH_PREDICTOR_PATH) or "your_specific_run_timestamp" in str(LENGTH_PREDICTOR_PATH)): # More general placeholder check
        logger.critical(f"CRITICAL ERROR: LENGTH_PREDICTOR_PATH is a placeholder or does not exist: {LENGTH_PREDICTOR_PATH}")
        logger.critical("Please update it to the actual path of your trained .pth file.")
        sys.exit(1) # Exit if critical path is not set
    
    # Create eval_output directory if it doesn't exist for result files
    eval_output_dir = Path("./eval_output") # Assuming template puts files here
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_length_predictor()