import json
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
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

RESULTS_JSONL_FILE = RESULTS_JSONL_FILE_LIST[1]
ERROR_PROFILE_OUTPUT_FILE = FIGURES_OUTPUT_FILE_LIST[1]
EVAL_OUTPUT_BASE_DIR = Path("./eval_output/clean")

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
                "mean_abs_pred_ratio": np.mean(np.abs(ratios)),
                "mean_abs_error_ratio": np.mean(np.abs(ratios) - 1.0),
                "percentiles": {
                    f"p{p}": np.percentile(ratios, p)
                    for p in [10, 25, 50, 75, 90]
                },
                "percentiles_abs_error": {
                    f"p{p}": np.percentile(np.abs(ratios) - 1.0, p)
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


def generate_error_profiles_all(jsonl_file_path: Path, output_file: Path):
    """
    Reads detailed step-by-step prediction results and generates a single error profile
    for all data combined, regardless of decoding parameters.
    """
    if not jsonl_file_path.exists():
        logger_analysis.error(f"Results file not found: {jsonl_file_path}")
        return
    
    # Collect all errors and ratios in simple lists
    all_errors = []
    all_ratios = []
    all_prompt_ids = set()
    
    logger_analysis.info(f"Processing results from {jsonl_file_path} to generate combined error profile...")
    num_sessions_processed = 0
    num_valid_initial_predictions = 0
    
    with open(jsonl_file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            try:
                session_result = json.loads(line)
                num_sessions_processed += 1
                
                step_predictions = session_result.get("step_predictions")
                prompt_tokenized_len = session_result.get("prompt_tokenized_len")
                actual_generated_steps_session = session_result.get("actual_generated_steps")
                prompt_id = session_result.get("prompt_id")
                
                if not isinstance(step_predictions, list) or \
                    not step_predictions or \
                    prompt_tokenized_len is None or \
                    actual_generated_steps_session is None:
                    continue
                
                # Get the prediction made at step_index = 0 (immediately after prefill)
                initial_prediction_step = None
                for step_data in step_predictions:
                    if isinstance(step_data, dict) and step_data.get("step_index") == 0:
                        if step_data.get("current_full_sequence_len_for_pred") == prompt_tokenized_len:
                            initial_prediction_step = step_data
                            break
                
                if initial_prediction_step and "predicted_rest_len" in initial_prediction_step:
                    num_valid_initial_predictions += 1
                    predicted_rest_at_prefill = initial_prediction_step["predicted_rest_len"]
                    
                    # Calculate initial prediction error
                    initial_error = predicted_rest_at_prefill - actual_generated_steps_session
                    
                    # Calculate initial error ratio
                    predicted_total_len = prompt_tokenized_len + predicted_rest_at_prefill
                    actual_total_len = prompt_tokenized_len + actual_generated_steps_session
                    
                    initial_error_ratio = np.nan
                    if actual_total_len > 0:
                        initial_error_ratio = predicted_total_len / actual_total_len
                    else:
                        if predicted_total_len == 0: initial_error_ratio = 1.0
                    
                    all_errors.append(initial_error)
                    if pd.notna(initial_error_ratio):
                        all_ratios.append(initial_error_ratio)
                    all_prompt_ids.add(prompt_id)
            
            except json.JSONDecodeError:
                logger_analysis.warning(f"Skipping invalid JSON line {line_idx+1} in {jsonl_file_path}")
            except Exception as e:
                logger_analysis.error(f"Error processing line {line_idx+1}: {e}")
    
    logger_analysis.info(f"Processed {num_sessions_processed} sessions, found {num_valid_initial_predictions} valid initial predictions.")
    
    # Calculate statistics for all data combined
    errors = np.array(all_errors)
    errors_abs = np.abs(errors)
    ratios = np.array(all_ratios)
    num_prompts = len(all_prompt_ids)
    
    if len(errors) == 0:
        logger_analysis.warning(f"No errors collected. Cannot generate profile.")
        return
    
    error_stats = {
        "mean_error": np.mean(errors),
        "median_error": np.median(errors),
        "std_dev_error": np.std(errors),
        "mae": np.mean(errors_abs),
        "rmse": np.sqrt(np.mean(errors**2)),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
        "percentiles": {
            f"p{p}": np.percentile(errors_abs, p)
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
            "mean_abs_pred_ratio": np.mean(np.abs(ratios)),
            "mean_abs_error_ratio": np.mean(np.abs(ratios) - 1.0),
            "percentiles": {
                f"p{p}": np.percentile(ratios, p)
                for p in [10, 25, 50, 75, 90]
            },
            "percentiles_abs_error": {
                f"p{p}": np.percentile(np.abs(ratios) - 1.0, p)
                for p in [10, 25, 50, 75, 90]
            }
        }
    else:
        ratio_stats = {k: np.nan for k in ["mean_pred_ratio", "median_pred_ratio", "std_dev_pred_ratio", "min_pred_ratio", "max_pred_ratio"]}
        ratio_stats["percentiles"] = {f"p{p}": np.nan for p in [10, 25, 50, 75, 90]}
    
    profile = {
        "profile_source": "combined_prefill_based_prediction (step_index=0)",
        "num_prompts_in_profile": num_prompts,
        "num_predictions": len(errors),
        "prediction_errors_stats": error_stats,
        "pred_ratios_stats": ratio_stats,
        "notes": "Statistics based on all predictions combined, regardless of decoding parameters."
    }
    
    # Save the error profile to a new JSON file (not JSONL since it's a single object)
    with open(output_file, 'w') as f_out_profile:
        json.dump(profile, f_out_profile, indent=2)
        
    logger_analysis.info(f"Generated combined error profile and saved to {output_file}")


def _plot_distribution_to_file(ratios_np: np.ndarray, title: str, output_path: Path,
                                num_prompts_involved: int = None, xlim_range: tuple = None):
    """Helper function to plot and save a single distribution with density."""
    if not isinstance(ratios_np, np.ndarray) or ratios_np.size == 0:
        logger_analysis.info(f"No data to plot for {title}. Skipping plot generation for: {output_path}")
        # Create a blank plot with "No data" text
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available for this distribution.",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Error Ratio (%)  [(Actual - Predicted) / Actual] * 100", fontsize=12)
        ax.set_ylabel("Density (Frequency)", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        logger_analysis.info(f"Saved empty placeholder plot to {output_path}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter data based on xlim_range if provided
    if xlim_range is not None:
        filtered_data = ratios_np[(ratios_np >= xlim_range[0]) & (ratios_np <= xlim_range[1])]
        if len(filtered_data) == 0:
            logger_analysis.warning(f"No data points in range {xlim_range} for {title}")
            filtered_data = ratios_np  # Fall back to all data
    else:
        filtered_data = ratios_np
    
    # Calculate frequency density: count / total_count for each bin
    # This gives the proportion/frequency of data in each bin
    counts, bins, _ = ax.hist(filtered_data, bins=50, edgecolor='black', alpha=0.75,
                                color='skyblue', density=False)
    
    # Convert counts to frequency density (proportion of total)
    total_count = len(ratios_np)  # Use total data count, not just filtered
    frequency_density = counts / total_count
    
    # Clear the plot and redraw with frequency density
    ax.clear()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax.bar(bin_centers, frequency_density, width=bin_width * 0.8,
            edgecolor='black', alpha=0.75, color='skyblue')
    
    # Calculate statistics from all data (not just filtered)
    mean_val = np.mean(ratios_np)
    median_val = np.median(ratios_np)
    std_val = np.std(ratios_np)
    
    # Only show vertical lines if they're within the plot range
    if xlim_range is None or (xlim_range[0] <= mean_val <= xlim_range[1]):
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5,
                    label=f'Mean: {mean_val:.2f}%')
    if xlim_range is None or (xlim_range[0] <= median_val <= xlim_range[1]):
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5,
                    label=f'Median: {median_val:.2f}%')
    
    # Set title
    plot_title = title
    if num_prompts_involved is not None:
        plot_title += f"\n(Based on {len(ratios_np)} predictions from {num_prompts_involved} unique prompts)"
    else:
        plot_title += f"\n(Based on {len(ratios_np)} predictions)"
    
    ax.set_title(plot_title, fontsize=15, pad=20)
    ax.set_xlabel("Error Ratio (%)  [(Actual Rest - Predicted Rest) / Actual Rest] * 100", fontsize=12)
    ax.set_ylabel("Density (Frequency)", fontsize=12)
    
    # Set x-axis limits if specified
    if xlim_range is not None:
        ax.set_xlim(xlim_range)
    
    # Position legend and stats text to avoid overlap
    legend_elements = ax.get_legend_handles_labels()
    if legend_elements[0]:  # If there are legend items
        ax.legend(loc='upper left', fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add stats text - position on the right side, separated from legend
    stats_text = f"Total Count: {len(ratios_np)}\nStd Dev: {std_val:.2f}%"
    if xlim_range is not None:
        filtered_count = len(filtered_data)
        filtered_pct = (filtered_count / len(ratios_np)) * 100
        stats_text += f"\nIn Range: {filtered_count} ({filtered_pct:.1f}%)"
        # Debug info to verify frequency calculation
        stats_text += f"\nDenominator: {total_count}"
        stats_text += f"\nSum of freq: {frequency_density.sum():.3f}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger_analysis.info(f"Saved distribution plot to {output_path}")


def plot_improved_step_zero_error_distributions(jsonl_file_path: Path, base_output_dir: Path):
    """
    Improved version of step-0 error ratio distribution plotting with:
    1. Two overall plots: wide range (-1000% to 1000%) and focused range (-150% to 150%)
    2. Parameter group plots with focused range (-150% to 150%)
    3. Density plots instead of frequency
    4. Better legend positioning to avoid overlap
    """
    logger_analysis.info(f"Starting improved step-0 error ratio distribution analysis from {jsonl_file_path}")
    
    if not jsonl_file_path.exists():
        logger_analysis.error(f"JSONL file not found: {jsonl_file_path}")
        return
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    per_param_group_dir = base_output_dir / "per_param_group"  # Limited range (-150% to 150%)
    group_step0_plots_dir = base_output_dir / "group_step0_error_ratio_plots"  # Full range
    per_param_group_dir.mkdir(parents=True, exist_ok=True)
    group_step0_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Data collection
    all_error_ratios_percentage = []
    all_prompt_ids_overall = set()
    data_by_params = {}
    num_skipped_zero_actual_len = 0
    num_valid_initial_predictions_for_ratio = 0
    
    # Process JSONL file
    with open(jsonl_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger_analysis.warning(f"Failed to parse JSON on line {line_num}: {e}")
                continue
            
            # Extract step-0 data
            step_predictions = data.get("step_predictions", [])
            if not step_predictions:
                continue
            
            step_0_data = step_predictions[0]
            if step_0_data.get("step_index") != 0:
                continue
            
            # Extract values
            actual_rest = step_0_data.get("actual_rest_len", 0)
            predicted_rest = step_0_data.get("predicted_rest_len", 0)
            prompt_id = data.get("prompt_id", f"unknown_{line_num}")
            
            # Get system parameters
            system_params = data.get("system_params", {})
            params_key = tuple(sorted(system_params.items()))
            
            # Calculate error ratio
            if actual_rest > 0:
                error_ratio_perc = ((actual_rest - predicted_rest) / actual_rest) * 100.0
                
                # Add to overall data
                all_error_ratios_percentage.append(error_ratio_perc)
                all_prompt_ids_overall.add(prompt_id)
                num_valid_initial_predictions_for_ratio += 1
                
                # Add to parameter group data
                if params_key not in data_by_params:
                    data_by_params[params_key] = {
                        "error_ratios_percentage": [],
                        "prompt_ids": set(),
                        "params_dict": system_params
                    }
                
                data_by_params[params_key]["error_ratios_percentage"].append(error_ratio_perc)
                data_by_params[params_key]["prompt_ids"].add(prompt_id)
            
            elif abs(actual_rest) < 1e-9 and abs(predicted_rest) >= 0.5:
                num_skipped_zero_actual_len += 1
    
    logger_analysis.info(f"Found {num_valid_initial_predictions_for_ratio} valid initial predictions for error ratio calculation.")
    if num_skipped_zero_actual_len > 0:
        logger_analysis.info(f"Skipped {num_skipped_zero_actual_len} predictions where actual_rest_len was 0 and predicted_rest_len was significantly non-zero.")
    
    if not all_error_ratios_percentage:
        logger_analysis.warning("No valid error ratios found. Cannot generate plots.")
        return
    
    # Generate overall plots
    all_ratios_array = np.array(all_error_ratios_percentage)
    
    # Plot 1: Wide range (-1000% to 1000%)
    overall_plot_wide = base_output_dir / "overall_step0_error_ratio_distribution_wide.png"
    _plot_distribution_to_file(
        all_ratios_array,
        "Overall Step-0 Error Ratio Distribution (Wide Range)",
        overall_plot_wide,
        len(all_prompt_ids_overall),
        xlim_range=(-1000, 1000)
    )
    
    # Plot 2: Focused range (-150% to 150%)
    overall_plot_focused = base_output_dir / "overall_step0_error_ratio_distribution_focused.png"
    _plot_distribution_to_file(
        all_ratios_array,
        "Overall Step-0 Error Ratio Distribution (Focused Range)",
        overall_plot_focused,
        len(all_prompt_ids_overall),
        xlim_range=(-150, 150)
    )
    
    # Generate parameter group plots
    if not data_by_params:
        logger_analysis.info("No data collected for any specific parameter group. Skipping group plots.")
        return
    
    for params_key, data_dict in data_by_params.items():
        params_as_dict = data_dict["params_dict"]
        
        # Create filename
        filename_parts = []
        for k, v in params_as_dict.items():
            clean_k = str(k).replace('_', '').replace('temperature', 'T').replace('tokens', 'tok')
            filename_parts.append(f"{clean_k}{v}")
        filename_base = "_".join(filename_parts)
        
        # Hash if too long
        if len(filename_base) > 80:
            filename_base = hashlib.md5(str(params_as_dict).encode()).hexdigest()[:16]
        
        # Generate focused range plot for parameter group (-150% to 150%)
        focused_plot_filename = f"err_ratio_dist_{filename_base}_focused.png"
        focused_plot_path = per_param_group_dir / focused_plot_filename
        
        focused_title = f"Step-0 Error Ratio Distribution (Focused Range)\nParams: {params_as_dict}"
        num_prompts_in_group = len(data_dict["prompt_ids"])
        
        _plot_distribution_to_file(
            np.array(data_dict["error_ratios_percentage"]),
            focused_title,
            focused_plot_path,
            num_prompts_in_group,
            xlim_range=(-150, 150)
        )
        
        # Generate full range plot for parameter group (no range limit)
        full_plot_filename = f"err_ratio_dist_{filename_base}_full.png"
        full_plot_path = group_step0_plots_dir / full_plot_filename
        
        full_title = f"Step-0 Error Ratio Distribution (Full Range)\nParams: {params_as_dict}"
        
        _plot_distribution_to_file(
            np.array(data_dict["error_ratios_percentage"]),
            full_title,
            full_plot_path,
            num_prompts_in_group,
            xlim_range=None  # No range limit
        )
    
    logger_analysis.info(f"Focused range plots saved in {per_param_group_dir}")
    logger_analysis.info(f"Full range plots saved in {group_step0_plots_dir}")
    logger_analysis.info("Finished improved step-0 error ratio distribution plotting.")


def plot_step_zero_error_ratio_distributions(jsonl_file_path: Path, base_output_dir: Path):
    """
    Reads prediction results, calculates step-0 error ratios, and plots their
    distributions (overall and per-parameter group) into the base_output_dir.
    Error Ratio (%) = (actual_rest_len - predicted_rest_len) / actual_rest_len * 100
    """
    if not jsonl_file_path.exists():
        logger_analysis.error(f"Results file not found: {jsonl_file_path}")
        return
    
    # Ensure base_output_dir and subdirectory for group plots exist
    base_output_dir.mkdir(parents=True, exist_ok=True)
    group_plots_dir = base_output_dir / "group_step0_error_ratio_plots"
    group_plots_dir.mkdir(parents=True, exist_ok=True)
    
    data_by_params = defaultdict(lambda: {"error_ratios_percentage": [], "prompt_ids": set()})
    all_error_ratios_percentage = []
    all_prompt_ids_overall = set()
    
    logger_analysis.info(f"Processing results from {jsonl_file_path} for plotting step-0 error ratio distributions...")
    num_sessions_processed = 0
    num_valid_initial_predictions_for_ratio = 0
    num_skipped_zero_actual_len = 0
    
    with open(jsonl_file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            try:
                session_result = json.loads(line)
                num_sessions_processed += 1
                
                dec_params = session_result.get("decoding_params")
                step_predictions = session_result.get("step_predictions")
                prompt_tokenized_len = session_result.get("prompt_tokenized_len")
                prompt_id = session_result.get("prompt_id")
                
                if not isinstance(dec_params, dict) or \
                    not isinstance(step_predictions, list) or \
                    not step_predictions or \
                    prompt_tokenized_len is None:
                    continue
                
                initial_prediction_step_data = None
                for step_data in step_predictions:
                    if isinstance(step_data, dict) and step_data.get("step_index") == 0:
                        # Ensure this is the prediction right after prefill
                        if step_data.get("current_full_sequence_len_for_pred") == prompt_tokenized_len:
                            initial_prediction_step_data = step_data
                            break
                
                if initial_prediction_step_data and \
                    "predicted_rest_len" in initial_prediction_step_data and \
                    "actual_rest_len" in initial_prediction_step_data:
                    
                    predicted_rest = initial_prediction_step_data["predicted_rest_len"]
                    actual_rest = initial_prediction_step_data["actual_rest_len"]
                    
                    error_ratio_perc = np.nan
                    if actual_rest > 0:
                        # Error Ratio (%) = (Actual - Predicted) / Actual * 100
                        error_ratio_perc = ((actual_rest - predicted_rest) / actual_rest) * 100.0
                    elif abs(actual_rest) < 1e-9: # actual_rest is effectively zero
                        if abs(predicted_rest) < 0.5 : # Prediction is also (close to) zero
                            error_ratio_perc = 0.0
                        else:
                            # Actual is 0, but predicted non-zero. Percentage error is problematic.
                            # Could assign a large penalty or skip. Here, we skip.
                            num_skipped_zero_actual_len += 1
                            # logger_analysis.debug(
                            # f"Skipping error ratio for prompt {prompt_id}, params {dec_params} due to "
                            # f"actual_rest_len=0 and predicted_rest_len={predicted_rest:.2f}"
                            # )
                            continue 
                    
                    if not np.isnan(error_ratio_perc):
                        num_valid_initial_predictions_for_ratio += 1
                        param_key = frozenset(dec_params.items()) # Use frozenset for dict key
                        data_by_params[param_key]["error_ratios_percentage"].append(error_ratio_perc)
                        data_by_params[param_key]["prompt_ids"].add(prompt_id)
                        
                        all_error_ratios_percentage.append(error_ratio_perc)
                        all_prompt_ids_overall.add(prompt_id)
                # else:
                    # logger_analysis.debug(f"No valid step-0 prediction found matching criteria for session line {line_idx+1}")
            
            except json.JSONDecodeError:
                logger_analysis.warning(f"Skipping invalid JSON line {line_idx+1} in {jsonl_file_path}")
            except Exception as e:
                logger_analysis.error(f"Error processing line {line_idx+1} for error ratio plot: {e}")
    
    logger_analysis.info(f"Processed {num_sessions_processed} sessions for plotting.")
    logger_analysis.info(f"Found {num_valid_initial_predictions_for_ratio} valid initial predictions for error ratio calculation.")
    if num_skipped_zero_actual_len > 0:
        logger_analysis.info(f"Skipped {num_skipped_zero_actual_len} predictions where actual_rest_len was 0 and predicted_rest_len was significantly non-zero.")
    
    # --- Plot Overall Distribution with Multiple Ranges ---
    all_ratios_array = np.array(all_error_ratios_percentage)
    
    # Plot 1: Wide range (-1000% to 1000%)
    overall_plot_wide = base_output_dir / "overall_step0_error_ratio_distribution_wide.png"
    _plot_distribution_to_file(
        all_ratios_array,
        "Overall Step-0 Error Ratio Distribution (Wide Range)",
        overall_plot_wide,
        len(all_prompt_ids_overall),
        xlim_range=(-1000, 1000)
    )
    
    # Plot 2: Focused range (-150% to 150%)
    overall_plot_focused = base_output_dir / "overall_step0_error_ratio_distribution_focused.png"
    _plot_distribution_to_file(
        all_ratios_array,
        "Overall Step-0 Error Ratio Distribution (Focused Range)",
        overall_plot_focused,
        len(all_prompt_ids_overall),
        xlim_range=(-150, 150)
    )
    
    # --- Plot Per-Parameter Group Distributions ---
    if not data_by_params:
        logger_analysis.info("No data collected for any specific parameter group. Skipping group plots.")
    else:
        logger_analysis.info(f"Generating plots for {len(data_by_params)} unique decoding parameter groups.")
        
    # Sort for consistent output order if script is run multiple times
    sorted_param_groups = sorted(data_by_params.items(), key=lambda item: str(dict(item[0])))
    
    for i, (param_key, data_dict) in enumerate(sorted_param_groups):
        params_as_dict = dict(param_key)
        
        # Create a somewhat readable and safe filename from parameters
        param_str_for_fname = []
        for k, v in sorted(params_as_dict.items()):
            k_clean = str(k).replace('.', '').replace('_', '')
            v_clean = str(v).replace('.', '').replace('_', '')
            param_str_for_fname.append(f"{k_clean}{v_clean}")
        
        filename_base = "_".join(param_str_for_fname)
        if not filename_base: filename_base = f"group_{i}" # Fallback for empty params
        
        # To keep filenames manageable, hash if too long or contains tricky characters
        if len(filename_base) > 80: # Arbitrary length limit
            filename_base = hashlib.md5(str(params_as_dict).encode()).hexdigest()[:16]
            
        # Generate focused range plot for each parameter group (-150% to 150%)
        group_plot_filename = f"err_ratio_dist_{filename_base}_focused.png"
        group_plot_path = group_plots_dir / group_plot_filename
        
        group_title = f"Step-0 Error Ratio Distribution (Focused Range)\nParams: {params_as_dict}"
        num_prompts_in_group = len(data_dict["prompt_ids"])
        
        _plot_distribution_to_file(
            np.array(data_dict["error_ratios_percentage"]),
            group_title,
            group_plot_path,
            num_prompts_in_group,
            xlim_range=(-150, 150)
        )
    
    if data_by_params:
        logger_analysis.info(f"All group distribution plots saved in {group_plots_dir}")
    logger_analysis.info("Finished plotting step-0 error ratio distributions.")


if __name__ == '__main__':
    # This script is intended to be run after the main evaluation script
    # (evaluate_length_predictor.py) has produced the detailed JSONL results.
    
    if not RESULTS_JSONL_FILE.exists() or RESULTS_JSONL_FILE.stat().st_size == 0:
        logger_analysis.error(f"Input results file for generating error profiles not found or is empty: {RESULTS_JSONL_FILE}")
        logger_analysis.error("Please run the main evaluation script first or update the path.")
    else:
        # Generate error profiles
        # generate_prefill_error_profiles(RESULTS_JSONL_FILE, ERROR_PROFILE_OUTPUT_FILE)
        
        # Use improved plotting function with multiple ranges and density plots
        plot_improved_step_zero_error_distributions(RESULTS_JSONL_FILE, EVAL_OUTPUT_BASE_DIR)
        
        # Optionally also run original function for comparison
        plot_step_zero_error_ratio_distributions(RESULTS_JSONL_FILE, EVAL_OUTPUT_BASE_DIR)
