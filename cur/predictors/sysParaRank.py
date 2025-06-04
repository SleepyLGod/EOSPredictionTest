# rank_system_parameters.py

import json
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ranking_script")

# --- Configuration ---
INPUT_EVAL_JSONL_FILE_LIST = [
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250527_194051.jsonl"), # Databricks
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250526_151530.jsonl"), # Clean
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250529_000246.jsonl"), # Eval
]
OUTPUT_FILE_LIST = [
    Path("./eval_output/rank_param_databricks.txt"),
    Path("./eval_output/rank_param_clean.txt"),
    Path("./eval_output/rank_param_eval.txt"),
]

def aggregate_metrics_from_raw_logs(file_path: Path) -> list:
    """
    Loads raw evaluation logs, extracts step-0 predictions,
    and aggregates metrics for each unique decoding parameter group.
    """
    if not file_path.exists():
        logger.error(f"Input evaluation log file not found: {file_path}")
        return []
    
    # Structure: { frozenset(params.items()): {'errors': [], 'error_ratios': [], 'pred_ratios': [], 'num_prompts': 0, 'params_dict': {}} }
    data_by_params_group = defaultdict(lambda: {
        "raw_errors_step0": [],
        "error_ratios_step0": [], # (Actual - Predicted) / Actual * 100%
        "prediction_ratios_step0": [], # (PromptLen + Pred) / (PromptLen + Actual)
        "prompt_ids": set(),
        "params_dict": {}
    })
    
    logger.info(f"Processing raw evaluation logs from: {file_path}")
    num_lines_processed = 0
    num_sessions_with_step0_pred = 0
    
    with open(file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            num_lines_processed += 1
            try:
                session_result = json.loads(line)
                
                dec_params = session_result.get("decoding_params")
                step_predictions = session_result.get("step_predictions")
                prompt_tokenized_len = session_result.get("prompt_tokenized_len")
                # This is the total number of tokens generated in the session AFTER the prompt.
                actual_generated_tokens_in_session = session_result.get("actual_generated_steps")
                prompt_id = session_result.get("prompt_id")
                
                if not isinstance(dec_params, dict) or \
                    not isinstance(step_predictions, list) or \
                    not step_predictions or \
                    prompt_tokenized_len is None or \
                    actual_generated_tokens_in_session is None:
                    # logger.debug(f"Skipping line {line_idx+1} due to missing essential data.")
                    continue
                
                # Find the step_index = 0 prediction
                initial_prediction_data = None
                for step_data in step_predictions:
                    if isinstance(step_data, dict) and step_data.get("step_index") == 0:
                        # Sanity check: ensure this prediction was made based on the prompt only
                        if step_data.get("current_full_sequence_len_for_pred") == prompt_tokenized_len:
                            initial_prediction_data = step_data
                            break
                
                if initial_prediction_data and "predicted_rest_len" in initial_prediction_data:
                    num_sessions_with_step0_pred += 1
                    predicted_len_step0 = initial_prediction_data["predicted_rest_len"]
                    
                    # For step_index = 0, the 'actual_rest_len' is simply total tokens generated in the session
                    actual_len_step0 = actual_generated_tokens_in_session
                    
                    # 1. Raw Error
                    raw_error = predicted_len_step0 - actual_len_step0
                    
                    # 2. Error Ratio: (Actual - Predicted) / Actual * 100%
                    error_ratio_perc = np.nan
                    if actual_len_step0 > 0:
                        error_ratio_perc = ((actual_len_step0 - predicted_len_step0) / actual_len_step0) * 100.0
                    elif actual_len_step0 == 0: # Actual is 0
                        if abs(predicted_len_step0) < 0.5: # Predicted is also (close to) 0
                            error_ratio_perc = 0.0
                        # Else, predicted is non-zero, actual is zero. Ratio is undefined/infinite. Keep as NaN.
                    
                    # 3. Prediction Ratio: (PromptLen + PredictedRest) / (PromptLen + ActualRest)
                    pred_ratio = np.nan
                    actual_total_len = prompt_tokenized_len + actual_len_step0
                    if actual_total_len > 0:
                        predicted_total_len = prompt_tokenized_len + predicted_len_step0
                        pred_ratio = predicted_total_len / actual_total_len
                    elif actual_total_len == 0: # PromptLen and ActualRest are both 0
                            predicted_total_len = prompt_tokenized_len + predicted_len_step0
                            if predicted_total_len == 0: # PredictedRest is also 0
                                pred_ratio = 1.0 # Perfect prediction of total length 0
                            # Else, predicted non-zero total, actual total is zero. Keep as NaN.
                            
                    
                    param_key = frozenset(dec_params.items())
                    group_data = data_by_params_group[param_key]
                    
                    group_data["params_dict"] = dec_params # Store once
                    group_data["prompt_ids"].add(prompt_id)
                    group_data["raw_errors_step0"].append(raw_error)
                    if pd.notna(error_ratio_perc):
                        group_data["error_ratios_step0"].append(error_ratio_perc)
                    if pd.notna(pred_ratio):
                        group_data["prediction_ratios_step0"].append(pred_ratio)
            
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line {line_idx+1} in {file_path}")
            except Exception as e:
                logger.error(f"Error processing line {line_idx+1}: {e}")
    
    logger.info(f"Processed {num_lines_processed} lines. Found {num_sessions_with_step0_pred} sessions with valid step-0 predictions.")
    logger.info(f"Aggregated data for {len(data_by_params_group)} unique parameter groups.")
    
    # Now, calculate summary statistics for each group
    aggregated_stats_list = []
    for param_key, group_data in data_by_params_group.items():
        stats = {"decoding_params": group_data["params_dict"], "num_prompts": len(group_data["prompt_ids"])}
        
        # Raw Errors
        raw_errors_np = np.array(group_data["raw_errors_step0"])
        if raw_errors_np.size > 0:
            stats["MAE"] = np.mean(np.abs(raw_errors_np))
            stats["RMSE"] = np.sqrt(np.mean(np.square(raw_errors_np)))
            stats["Abs_Bias"] = np.abs(np.mean(raw_errors_np))
            stats["Std_Dev_Error"] = np.std(raw_errors_np)
        else: # Should not happen if group exists, but good for safety
            stats["MAE"] = stats["RMSE"] = stats["Abs_Bias"] = stats["Std_Dev_Error"] = np.nan
        
        # Error Ratios
        error_ratios_np = np.array(group_data["error_ratios_step0"])
        if error_ratios_np.size > 0:
            stats["Mean_Abs_Error_Ratio_Pct"] = np.mean(np.abs(error_ratios_np))
            stats["Std_Dev_Error_Ratio_Pct"] = np.std(error_ratios_np)
        else:
            stats["Mean_Abs_Error_Ratio_Pct"] = stats["Std_Dev_Error_Ratio_Pct"] = np.nan
            
        # Prediction Ratios
        pred_ratios_np = np.array(group_data["prediction_ratios_step0"])
        if pred_ratios_np.size > 0:
            stats["Mean_Abs_Pred_Ratio_Error_From_1"] = np.mean(np.abs(pred_ratios_np - 1.0))
            stats["Std_Dev_Pred_Ratio"] = np.std(pred_ratios_np)
        else:
            stats["Mean_Abs_Pred_Ratio_Error_From_1"] = stats["Std_Dev_Pred_Ratio"] = np.nan
            
        aggregated_stats_list.append(stats)
        
    return aggregated_stats_list


def rank_parameter_groups(aggregated_stats: list, top_n: int = 20) -> pd.DataFrame:
    """
    Ranks parameter groups based on a composite score from aggregated metrics.
    Lower is better for all individual metrics and the final score.
    """
    if not aggregated_stats:
        logger.warning("No aggregated_stats provided for ranking.")
        return pd.DataFrame()
    
    data_for_df = []
    large_penalty_value = 1e9 # A large number for penalty
    
    for profile_stats in aggregated_stats:
        # Ensure all expected keys are present, default to NaN then penalty
        row_data = {
            "decoding_params": str(profile_stats.get("decoding_params", {})),
            "num_prompts": profile_stats.get("num_prompts", 0),
            "MAE": profile_stats.get("MAE", np.nan),
            "RMSE": profile_stats.get("RMSE", np.nan),
            "Abs_Bias": profile_stats.get("Abs_Bias", np.nan),
            "Std_Dev_Error": profile_stats.get("Std_Dev_Error", np.nan),
            "Mean_Abs_Error_Ratio_Pct": profile_stats.get("Mean_Abs_Error_Ratio_Pct", np.nan),
            "Std_Dev_Error_Ratio_Pct": profile_stats.get("Std_Dev_Error_Ratio_Pct", np.nan),
            "Mean_Abs_Pred_Ratio_Error_From_1": profile_stats.get("Mean_Abs_Pred_Ratio_Error_From_1", np.nan),
            "Std_Dev_Pred_Ratio": profile_stats.get("Std_Dev_Pred_Ratio", np.nan),
            "_original_params": profile_stats.get("decoding_params", {})
        }
        data_for_df.append(row_data)
    
    df = pd.DataFrame(data_for_df)
    
    metrics_to_rank = [
        "MAE", "RMSE", "Abs_Bias", "Std_Dev_Error",
        "Mean_Abs_Error_Ratio_Pct", "Std_Dev_Error_Ratio_Pct",
        "Mean_Abs_Pred_Ratio_Error_From_1", "Std_Dev_Pred_Ratio"
    ]
    
    # Apply penalty for NaNs before ranking
    for metric_col in metrics_to_rank:
        df[metric_col] = df[metric_col].fillna(large_penalty_value)
    
    # Calculate ranks for each metric. Lower value = better rank (rank 1).
    for metric_col in metrics_to_rank:
        df[f"{metric_col}_Rank"] = df[metric_col].rank(method="min") # NaNs (now large_penalty_value) get worst rank
    
    # Calculate composite score: sum of ranks. Lower sum is better.
    rank_cols = [f"{metric_col}_Rank" for metric_col in metrics_to_rank]
    df["Composite_Score"] = df[rank_cols].sum(axis=1)
    
    # Sort by composite score then by MAE as a tie-breaker
    df_sorted = df.sort_values(by=["Composite_Score", "MAE"], ascending=[True, True]).reset_index(drop=True)
    
    # Select top N
    top_n_df = df_sorted.head(top_n)
    
    logger.info(f"\n--- Top {top_n} System Parameter Groups (ranked from raw logs) ---")
    for index, row in top_n_df.iterrows():
        logger.info(f"Rank {index + 1}: Score = {row['Composite_Score']:.2f}, Params = {row['decoding_params']}")
        details = []
        for metric in metrics_to_rank:
            if row[metric] < large_penalty_value * 0.9 : # Don't show penalty values
                details.append(f"{metric}={row[metric]:.3f} (R:{row[f'{metric}_Rank']:.0f})")
            else:
                details.append(f"{metric}=N/A (R:{row[f'{metric}_Rank']:.0f})")
        
        # Print metrics in chunks for readability
        chunk_size = 3
        for i in range(0, len(details), chunk_size):
            logger.info(f"  Metrics: {', '.join(details[i:i+chunk_size])}")
        logger.info(f"  (Based on {row['num_prompts']} prompts)")
        logger.info("-" * 40)
        
    return top_n_df


def save_top_parameters_to_file(top_parameters_df: pd.DataFrame, output_file_path: Path, top_n: int = 20):
    """
    Save the top N parameter groups to a text file with detailed information.
    """
    # Ensure output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    large_penalty_value = 1e9
    metrics_to_rank = [
        "MAE", "RMSE", "Abs_Bias", "Std_Dev_Error",
        "Mean_Abs_Error_Ratio_Pct", "Std_Dev_Error_Ratio_Pct",
        "Mean_Abs_Pred_Ratio_Error_From_1", "Std_Dev_Pred_Ratio"
    ]
    
    with open(output_file_path, 'w') as f:
        f.write(f"Top {top_n} System Parameter Groups (ranked from raw logs)\n")
        f.write("=" * 80 + "\n\n")
        
        for index, row in top_parameters_df.iterrows():
            f.write(f"Rank {index + 1}: Score = {row['Composite_Score']:.2f}\n")
            f.write(f"Parameters: {row['decoding_params']}\n")
            f.write(f"Number of prompts: {row['num_prompts']}\n")
            
            # Write metrics details
            f.write("Metrics:\n")
            for metric in metrics_to_rank:
                if row[metric] < large_penalty_value * 0.9:  # Don't show penalty values
                    f.write(f"  {metric}: {row[metric]:.3f} (Rank: {row[f'{metric}_Rank']:.0f})\n")
                else:
                    f.write(f"  {metric}: N/A (Rank: {row[f'{metric}_Rank']:.0f})\n")

            f.write("-" * 80 + "\n\n")
    
    logger.info(f"Top {top_n} ranked parameters saved to {output_file_path}")


if __name__ == "__main__":
    logger.info("Starting system parameter ranking process from raw evaluation logs...")
    
    # Check if input and output lists have the same length
    if len(INPUT_EVAL_JSONL_FILE_LIST) != len(OUTPUT_FILE_LIST):
        logger.error(f"Mismatch between input files ({len(INPUT_EVAL_JSONL_FILE_LIST)}) and output files ({len(OUTPUT_FILE_LIST)})")
        exit(1)
    
    for input_file_path, output_file_path in zip(INPUT_EVAL_JSONL_FILE_LIST, OUTPUT_FILE_LIST):
        logger.info(f"Processing input file: {input_file_path}")
        logger.info(f"Output will be saved to: {output_file_path}")
        
        if not input_file_path.exists():
            logger.error(f"Input file not found: {input_file_path}")
            continue
        
        aggregated_stats_list = aggregate_metrics_from_raw_logs(input_file_path)
        
        if aggregated_stats_list:
            top_parameters_df = rank_parameter_groups(aggregated_stats_list, top_n=20)
            
            if not top_parameters_df.empty:
                save_top_parameters_to_file(top_parameters_df, output_file_path, top_n=20)
            else:
                logger.warning(f"No parameters to rank for file: {input_file_path}")
        else:
            logger.error(f"No data was aggregated from logs: {input_file_path}")
    
    logger.info("Ranking process complete.")