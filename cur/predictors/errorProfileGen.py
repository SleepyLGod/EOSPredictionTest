import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict # For easier aggregation

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger_analysis = logging.getLogger("analysis_script")

# --- Configurations ---
RESULTS_JSONL_FILE_LIST = [
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250527_194051.jsonl"), # Databricks
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250526_151530.jsonl"), # Clean
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250529_000246.jsonl"), # Eval
]
FIGURES_OUTPUT_FILE_LIST = [
    Path("./eval_output/prefill_err_profiles_databricks.jsonl"),
    Path("./eval_output/prefill_err_profiles_clean.jsonl"),
    Path("./eval_output/prefill_err_profiles_eval.jsonl"),
]

RESULTS_JSONL_FILE = RESULTS_JSONL_FILE_LIST[2]
ERROR_PROFILE_OUTPUT_FILE = FIGURES_OUTPUT_FILE_LIST[2]

ERROR_PROFILE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def generate_prefill_error_profiles(jsonl_file_path: Path, output_file: Path):
    """
    Reads detailed step-by-step prediction results and generates an error profile
    for the initial prediction made after the prefill stage (step_index = 0)
    for each unique set of decoding parameters.
    """
    if not jsonl_file_path.exists():
        logger_analysis.error(f"Results file not found: {jsonl_file_path}")
        return
    
    # Use defaultdict to collect errors and ratios per decoding_params configuration
    # The key will be a frozenset of decoding_params items to make it hashable
    # The value will be a dict {'errors': [], 'ratios': [], 'prompt_ids': set()}
    data_by_params = defaultdict(lambda: {"errors": [], "ratios": [], "prompt_ids": set()})
    
    logger_analysis.info(f"Processing results from {jsonl_file_path} to generate prefill error profiles...")
    num_sessions_processed = 0
    num_valid_initial_predictions = 0
    
    with open(jsonl_file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            try:
                session_result = json.loads(line)
                num_sessions_processed += 1
                
                dec_params = session_result.get("decoding_params")
                step_predictions = session_result.get("step_predictions")
                prompt_tokenized_len = session_result.get("prompt_tokenized_len")
                actual_generated_steps_session = session_result.get("actual_generated_steps")
                prompt_id = session_result.get("prompt_id") # Or original_base_prompt_id if using repeated runs
                
                if not isinstance(dec_params, dict) or \
                    not isinstance(step_predictions, list) or \
                    not step_predictions or \
                    prompt_tokenized_len is None or \
                    actual_generated_steps_session is None:
                    # logger_analysis.debug(f"Skipping session on line {line_idx+1} due to missing essential data.")
                    continue
                
                # Get the prediction made at step_index = 0 (immediately after prefill)
                initial_prediction_step = None
                for step_data in step_predictions:
                    if isinstance(step_data, dict) and step_data.get("step_index") == 0:
                        # Check if current_full_sequence_len_for_pred matches prompt_tokenized_len
                        if step_data.get("current_full_sequence_len_for_pred") == prompt_tokenized_len:
                            initial_prediction_step = step_data
                            break
                
                if initial_prediction_step and "predicted_rest_len" in initial_prediction_step:
                    num_valid_initial_predictions += 1
                    predicted_rest_at_prefill = initial_prediction_step["predicted_rest_len"]
                    
                    # Calculate initial prediction error
                    # Error = Predicted_Rest_After_Prompt - Actual_Generated_After_Prompt
                    initial_error = predicted_rest_at_prefill - actual_generated_steps_session
                    
                    # Calculate initial error ratio
                    # Ratio = (PromptLen + PredRest) / (PromptLen + ActualGenerated)
                    predicted_total_len = prompt_tokenized_len + predicted_rest_at_prefill
                    actual_total_len = prompt_tokenized_len + actual_generated_steps_session
                    
                    initial_error_ratio = np.nan # Default to NaN
                    if actual_total_len > 0: # Avoid division by zero
                        initial_error_ratio = predicted_total_len / actual_total_len
                    else: # This case should be rare if prompt_tokenized_len > 0
                        if predicted_total_len == 0: initial_error_ratio = 1.0 # Both zero, consider perfect
                        # else, it's problematic, NaN is fine
                    
                    # Use frozenset of items for dict key
                    param_key = frozenset(dec_params.items())
                    data_by_params[param_key]["errors"].append(initial_error)
                    if pd.notna(initial_error_ratio): # Only append valid ratios
                        data_by_params[param_key]["ratios"].append(initial_error_ratio)
                    data_by_params[param_key]["prompt_ids"].add(prompt_id) # Count unique prompts for this profile
                # else:
                    # logger_analysis.debug(f"No valid initial prediction (step_index=0, matching prompt_len) found for session on line {line_idx+1}.")
            
            except json.JSONDecodeError:
                logger_analysis.warning(f"Skipping invalid JSON line {line_idx+1} in {jsonl_file_path}")
            except Exception as e:
                logger_analysis.error(f"Error processing line {line_idx+1}: {e}")
    
    logger_analysis.info(f"Processed {num_sessions_processed} sessions, found {num_valid_initial_predictions} valid initial predictions.")
    
    # --- Now, for each parameter group, calculate statistics and write profile ---
    error_profiles = []
    for param_key, data_dict in data_by_params.items():
        errors = np.array(data_dict["errors"])
        errors_abs = np.abs(errors)
        ratios = np.array(data_dict["ratios"]) # Might be empty if all actual_total_len were 0
        num_prompts = len(data_dict["prompt_ids"])
        
        if len(errors) == 0: # Should not happen if num_valid_initial_predictions > 0 and data_by_params was populated
            logger_analysis.warning(f"No errors collected for param group {dict(param_key)}. Skipping profile.")
            continue
        
        error_stats = {
            "mean_error": np.mean(errors) if len(errors) > 0 else np.nan,
            "median_error": np.median(errors) if len(errors) > 0 else np.nan,
            "std_dev_error": np.std(errors) if len(errors) > 0 else np.nan,
            "mae": np.mean(errors_abs) if len(errors_abs) > 0 else np.nan,
            "rmse": np.sqrt(np.mean(errors**2)) if len(errors) > 0 else np.nan,
            "min_error": np.min(errors) if len(errors) > 0 else np.nan,
            "max_error": np.max(errors) if len(errors) > 0 else np.nan,
            "percentiles": {
                f"p{p}": np.percentile(errors_abs, p) if len(errors_abs) > 0 else np.nan
                for p in [10, 25, 50, 75, 90]
            }
        }
        
        ratio_stats = {}
        if len(ratios) > 0:
            ratio_stats = {
                "mean_pred_ratio": np.mean(ratios),
                "median_pred_ratio": np.median(ratios),
                "std_dev_pred_ratio": np.std(ratios),
                "min_pred_ratio": np.min(ratios),
                "max_pred_ratio": np.max(ratios),
                "percentiles": {
                    f"p{p}": np.percentile(ratios, p)
                    for p in [10, 25, 50, 75, 90]
                }
            }
        else: # Fill with NaN if no valid ratios
            ratio_stats = {k: np.nan for k in ["mean_pred_ratio", "median_pred_ratio", "std_dev_pred_ratio", "min_pred_ratio", "max_pred_ratio"]}
            ratio_stats["percentiles"] = {f"p{p}": np.nan for p in [10, 25, 50, 75, 90]}
        
        profile = {
            "decoding_params": dict(param_key), # Convert frozenset back to dict for JSON
            "profile_source": "prefill_based_prediction (step_index=0)",
            "num_prompts_in_profile": num_prompts,
            "prediction_errors_stats": error_stats,
            "pred_ratios_stats": ratio_stats,
            "notes": "Statistics based on the single prediction made immediately after prompt prefilling."
        }
        error_profiles.append(profile)
    
    # Save the error profiles to a new JSONL file
    with open(output_file, 'w') as f_out_profile:
        for profile_entry in error_profiles:
            f_out_profile.write(json.dumps(profile_entry) + '\n')
            
    logger_analysis.info(f"Generated {len(error_profiles)} prefill error profiles and saved to {output_file}")


if __name__ == '__main__':
    # This script is intended to be run after the main evaluation script
    # (evaluate_length_predictor.py) has produced the detailed JSONL results.
    
    if not RESULTS_JSONL_FILE.exists() or RESULTS_JSONL_FILE.stat().st_size == 0:
        logger_analysis.error(f"Input results file for generating error profiles not found or is empty: {RESULTS_JSONL_FILE}")
        logger_analysis.error("Please run the main evaluation script first or update the path.")
    else:
        generate_prefill_error_profiles(RESULTS_JSONL_FILE, ERROR_PROFILE_OUTPUT_FILE)