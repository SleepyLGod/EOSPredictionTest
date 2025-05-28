import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import itertools # For iterating parameter groups
from typing import Optional,List, Dict, Any # For type hinting
from tqdm import tqdm # For progress bars
from sklearn.metrics import r2_score

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger_analysis = logging.getLogger("analysis_script")

# --- Configurations ---
RESULTS_JSONL_FILE = Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250526_151530.jsonl")
FIGURES_OUTPUT_DIR = Path("./results/clean/")
METADATA_OUTPUT_FILE = FIGURES_OUTPUT_DIR / "aggregated_metadata_and_stats.json"

FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These MIN/MAX constants MUST match those used for normalization when training the length predictor
# and cover the ranges in your DECODING_PARAMS_LIST used in the evaluation script.
MIN_TEMP, MAX_TEMP = 0.1, 0.9
MIN_TOP_K, MAX_TOP_K = 1, 100
MIN_REP_PENALTY, MAX_REP_PENALTY = 1.3, 1.6 # Adjusted based on previous discussions
MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS = 100, 500 # Adjusted based on previous discussions
MIN_SEQ_POS, MAX_SEQ_POS = 0, 8191 # Should match training normalization range for seq_pos

# New constant for tail analysis
TAIL_LENGTH_THRESHOLD = 5 # Steps where actual_rest_len <= this are considered "tail"

# --- Helper NpEncoder for JSON serialization ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, pd.Interval): return str(obj) # Handle pandas Interval for bin labels
        return super(NpEncoder, self).default(obj)

# --- Function to load and preprocess data (same as your last version) ---
def load_results_to_dataframe(jsonl_file_path: Path) -> pd.DataFrame:
    # ... (Keep the robust version from your last provided code) ...
    # ... (Ensure 'error_ratio' is calculated: 
    # df['predicted_total_len_from_step'] = df['current_full_sequence_len_for_pred'] + df['predicted_rest_len']
    # df['actual_total_len_from_step'] = df['current_full_sequence_len_for_pred'] + df['actual_rest_len']
    # df['error_ratio'] = df['predicted_total_len_from_step'] / df['actual_total_len_from_step'].replace(0, np.nan)
    # And 'abs_error': df['abs_error'] = df['prediction_error'].abs()
    # ) ...
    if not jsonl_file_path.exists():
        logger_analysis.error(f"Results file not found: {jsonl_file_path}")
        return pd.DataFrame()
    all_steps_data: List[Dict[str, Any]] = []
    with open(jsonl_file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            try:
                session_result = json.loads(line)
                prompt_id = session_result.get("prompt_id")
                dec_params = session_result.get("decoding_params", {})
                for step_detail in session_result.get("step_predictions", []):
                    flat_step = {
                        "prompt_id": prompt_id,
                        "temp": dec_params.get("temperature"), "top_k": dec_params.get("top_k"),
                        "rep_p": dec_params.get("repetition_penalty"), "max_new_tok_session": dec_params.get("max_new_tokens"),
                        "actual_generated_steps_session": session_result.get("actual_generated_steps"),
                        "eos_encountered_session": session_result.get("eos_encountered_in_session"),
                        **step_detail}
                    all_steps_data.append(flat_step)
            except json.JSONDecodeError: logger_analysis.warning(f"Skipping invalid JSON line {line_idx+1}")
            except Exception as e: logger_analysis.error(f"Error processing line {line_idx+1}: {e}")
    if not all_steps_data: return pd.DataFrame()
    df = pd.DataFrame(all_steps_data)
    logger_analysis.info(f"Loaded {len(df)} step records from {len(df['prompt_id'].unique()) if 'prompt_id' in df else 0} sessions.")
    numeric_cols = ['predicted_rest_len', 'actual_rest_len', 'latency_ms', 'prediction_error',
                    'current_full_sequence_len_for_pred', 'step_index', 'temp', 'top_k', 'rep_p',
                    'max_new_tok_session', 'actual_generated_steps_session']
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'current_full_sequence_len_for_pred' in df.columns and \
        'predicted_rest_len' in df.columns and 'actual_rest_len' in df.columns:
        df['predicted_total_len_from_step'] = df['current_full_sequence_len_for_pred'] + df['predicted_rest_len']
        df['actual_total_len_from_step'] = df['current_full_sequence_len_for_pred'] + df['actual_rest_len']
        df['error_ratio'] = df['predicted_total_len_from_step'] / df['actual_total_len_from_step'].replace(0, np.nan)
    if 'prediction_error' in df.columns: df['abs_error'] = df['prediction_error'].abs()
    essential_cols = ['prediction_error', 'actual_rest_len', 'latency_ms', 'temp', 'top_k', 'rep_p', 'max_new_tok_session']
    if 'error_ratio' in df.columns: essential_cols.append('error_ratio')
    df.dropna(subset=essential_cols, inplace=True)
    logger_analysis.info(f"DataFrame shape after processing: {df.shape}")
    return df


# --- Helper function to save plots (same as before) ---
def save_plot(plt_figure, fig_title: str, output_dir: Path, plot_name: str, is_subfolder: bool = False):
    """Helper to save plots and close them, supports subfolders."""
    # fig_title is now the main title for the figure, not suptitle for FacetGrids
    if not is_subfolder: # Only add suptitle if it's not a FacetGrid (which handles its own titles)
        plt_figure.suptitle(fig_title, fontsize=16, y=1.0) # Adjust y if needed
    
    plt_figure.tight_layout(rect=[0, 0, 1, 0.95 if not is_subfolder else 1]) # Adjust rect for suptitle
    
    final_path = output_dir / plot_name
    final_path.parent.mkdir(parents=True, exist_ok=True) # Ensure subfolder exists
    
    plt_figure.savefig(final_path)
    plt.close(plt_figure)
    logger_analysis.info(f"Generated and saved: {final_path}")


# --- Function for TRAIL-like heatmap (same as before) ---
def plot_length_prediction_heatmap(
    df: pd.DataFrame, output_dir: Path, filename: str, title: str,
    max_length_for_binning: int = 512, num_bins: int = 10,
    param_group_str: Optional[str] = None # For filename/title if per group
):
    # ... (Keep the robust version from your last provided code)
    # ... (It should use the passed 'title' and 'filename' directly)
    # ... (The save_plot call within this function should be: 
    #      save_plot(fig, title, output_dir, f"{filename}.png"))
    if df.empty: return
    if not all(p in df.columns for p in ['actual_rest_len', 'predicted_rest_len']): return
    df_heatmap = df.copy()
    df_heatmap['actual_rest_len_clipped'] = np.clip(pd.to_numeric(df_heatmap['actual_rest_len'], errors='coerce'), 0, max_length_for_binning)
    df_heatmap['predicted_rest_len_clipped'] = np.clip(pd.to_numeric(df_heatmap['predicted_rest_len'], errors='coerce'), 0, max_length_for_binning)
    df_heatmap.dropna(subset=['actual_rest_len_clipped', 'predicted_rest_len_clipped'], inplace=True)
    if df_heatmap.empty: return
    
    bin_edges = np.linspace(0, max_length_for_binning, num_bins + 1)
    bin_labels = [f'b{i+1}' for i in range(num_bins)]
    df_heatmap['actual_bin'] = pd.cut(df_heatmap['actual_rest_len_clipped'], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)
    df_heatmap['predicted_bin'] = pd.cut(df_heatmap['predicted_rest_len_clipped'], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)
    df_heatmap.dropna(subset=['actual_bin', 'predicted_bin'], inplace=True)
    if df_heatmap.empty: return
    
    contingency_table = pd.crosstab(df_heatmap['predicted_bin'], df_heatmap['actual_bin'])
    cat_type = pd.CategoricalDtype(categories=bin_labels, ordered=True)
    contingency_table.index = contingency_table.index.astype(cat_type); contingency_table.columns = contingency_table.columns.astype(cat_type)
    contingency_table = contingency_table.sort_index(axis=0).sort_index(axis=1)
    log_contingency_table = np.log1p(contingency_table)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(log_contingency_table, annot=False, fmt=".1f", cmap="Blues", linewidths=.5, cbar_kws={'label': 'Log(Count + 1)'}, ax=ax)
    ax.set_title(title if param_group_str is None else f"{title}\n({param_group_str})", fontsize=14)
    ax.set_xlabel("Groundtruth Remaining Length (Binned)"); ax.set_ylabel("Predicted Remaining Length (Binned)")
    save_plot(fig, title, output_dir, f"{filename}{'_'+param_group_str if param_group_str else ''}.png", is_subfolder=True)


# --- Main Analysis and Plotting Function ---
def analyze_and_plot_results(df: pd.DataFrame, base_output_dir: Path, metadata_store: dict):
    if df.empty:
        logger_analysis.info("DataFrame is empty, no analysis to perform.")
        return
    
    param_cols = ['temp', 'top_k', 'rep_p', 'max_new_tok_session']
    
    # --- 1. Overall Performance Metrics & Distributions (All Prompts, All Params) ---
    overall_dir = base_output_dir / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
    logger_analysis.info("\n--- Overall Performance Metrics (All Data) ---")
    
    overall_metrics = {}
    # MAE, RMSE, Bias for prediction_error
    if 'prediction_error' in df.columns:
        df['abs_error'] = df['prediction_error'].abs() # Calculate once
        overall_metrics['mae_tokens'] = df['abs_error'].mean()
        overall_metrics['rmse_tokens'] = np.sqrt((df['prediction_error']**2).mean())
        overall_metrics['bias_tokens'] = df['prediction_error'].mean()
        # ... log these ...
        fig_err_dist, ax_err_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df['prediction_error'], bins=50, kde=True, ax=ax_err_dist)
        save_plot(fig_err_dist, "Overall: Distribution of Prediction Error (tokens)", overall_dir, "overall_prediction_error_dist.png")
    
    # Total Output Length Prediction Ratio
    if 'error_ratio' in df.columns: # This is actually (PredTotal/ActualTotal)
        overall_metrics['mean_total_len_pred_ratio'] = df['error_ratio'].mean()
        overall_metrics['median_total_len_pred_ratio'] = df['error_ratio'].median()
        # ... log these ...
        fig_er_dist, ax_er_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df['error_ratio'].dropna().clip(0, 3), bins=50, kde=True, ax=ax_er_dist) # Clip for viz
        ax_er_dist.axvline(1.0, color='red', linestyle='--', label='Ideal Ratio (1.0)')
        ax_er_dist.legend()
        save_plot(fig_er_dist, "Overall: Distribution of Total Output Length Prediction Ratio (PredTotal/ActualTotal, Clipped)", overall_dir, "overall_total_len_pred_ratio_dist.png")
    
    # Relative Error Ratios for Remaining Length (Metrics 5 & 6 from user, with filtering)
    # Filter out actual_rest_len <= 2 before calculating these specific ratios
    df_rel_err = df[df['actual_rest_len'] > 2].copy() # Use .copy() to avoid SettingWithCopyWarning
    if not df_rel_err.empty and 'actual_rest_len' in df_rel_err and 'prediction_error' in df_rel_err:
        # Add a small epsilon to avoid division by zero, though >2 check mostly handles it
        epsilon = 1e-6 
        df_rel_err['rel_error_signed'] = df_rel_err['prediction_error'] / (df_rel_err['actual_rest_len'] + epsilon) # Error relative to actual remaining
        df_rel_err['rel_error_abs'] = df_rel_err['rel_error_signed'].abs()
        overall_metrics['mean_rel_error_signed_filtered'] = df_rel_err['rel_error_signed'].mean()
        overall_metrics['mean_rel_error_abs_filtered'] = df_rel_err['rel_error_abs'].mean()
        # ... log these ...
        fig_rel_err_dist, ax_rel_err_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df_rel_err['rel_error_abs'].dropna().clip(0, 2), bins=50, kde=True, ax=ax_rel_err_dist) # Clip for viz
        save_plot(fig_rel_err_dist, "Overall: Distribution of Absolute Relative Error on Remaining Length (ActualRestLen > 2, Clipped)", overall_dir, "overall_abs_rel_error_dist.png")
    
    # Tail Error Ratio Analysis (Metric 7)
    df_tail = df[df['actual_rest_len'] <= TAIL_LENGTH_THRESHOLD].copy()
    if not df_tail.empty and 'actual_rest_len' in df_tail and 'prediction_error' in df_tail:
        # Use the same definition as rel_error_signed but on the tail data
        epsilon = 1e-6
        df_tail['tail_error_ratio_signed'] = df_tail['prediction_error'] / (df_tail['actual_rest_len'] + epsilon)
        df_tail['tail_error_ratio_abs'] = df_tail['tail_error_ratio_signed'].abs()
        overall_metrics['mean_tail_abs_error_ratio'] = df_tail['tail_error_ratio_abs'].mean(skipna=True) # Skipna if epsilon made some NaN
        overall_metrics['median_tail_abs_error_ratio'] = df_tail['tail_error_ratio_abs'].median(skipna=True)
        logger_analysis.info(f"Overall Mean Absolute Tail Error Ratio (ActualRestLen <= {TAIL_LENGTH_THRESHOLD}): {overall_metrics.get('mean_tail_abs_error_ratio', 'N/A'):.3f}")
        fig_tail_err_dist, ax_tail_err_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df_tail['tail_error_ratio_abs'].dropna().clip(0,5), bins=30, kde=True, ax=ax_tail_err_dist) # Clip for viz
        save_plot(fig_tail_err_dist, f"Overall: Distribution of Absolute Tail Error Ratio (ActualRestLen <= {TAIL_LENGTH_THRESHOLD}, Clipped)", overall_dir, "overall_tail_abs_error_ratio_dist.png")
    
    # R-squared Score (Overall)
    if 'actual_rest_len' in df.columns and 'predicted_rest_len' in df.columns:
        # Ensure no NaNs in the specific columns for r2_score
        df_r2 = df.dropna(subset=['actual_rest_len', 'predicted_rest_len'])
        if len(df_r2) > 1: # r2_score needs at least 2 samples
            overall_metrics['r2_score_remaining_len'] = r2_score(df_r2['actual_rest_len'], df_r2['predicted_rest_len'])
            logger_analysis.info(f"Overall R-squared (R²) for Remaining Length: {overall_metrics['r2_score_remaining_len']:.3f}")
        else:
            logger_analysis.warning("Not enough data points to calculate overall R-squared score.")
            overall_metrics['r2_score_remaining_len'] = np.nan
    
    
    metadata_store['overall_performance'] = overall_metrics
    logger_analysis.info(f"Overall MAE tokens: {overall_metrics.get('mae_tokens', 'N/A'):.2f}") # Example log
    
    # Overall Latency (Violin Plot as before)
    if 'latency_ms' in df.columns:
        fig_lat_overall, ax_lat_overall = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, y='latency_ms', ax=ax_lat_overall, cut=0)
        ax_lat_overall.set_yscale('log')
        ax_lat_overall.set_title("Overall Distribution of Predictor Latency")
        save_plot(fig_lat_overall, "Overall Predictor Latency Distribution", overall_dir, "overall_latency_violin.png")
    
    
    # --- Per Parameter Group Analysis & Plots ---
    logger_analysis.info("\n--- Performance Metrics per System Parameter Group ---")
    # Get unique parameter groups
    # Convert list of dicts (decoding_params in results) to tuple of tuples for grouping
    # This was done when creating columns temp, top_k, rep_p, max_new_tok_session
    
    unique_param_groups = df[param_cols].drop_duplicates().to_dict('records')
    metadata_store['performance_per_param_group'] = []
    
    for i, group_params_dict in enumerate(tqdm(unique_param_groups, desc="Analyzing Parameter Groups")):
        # Create a string representation for filenames and titles
        group_params_str = "_".join([f"{k.replace('_','').replace('temperature','T').replace('tokens','tok')}{v}" for k, v in group_params_dict.items()])
        group_output_dir = base_output_dir / "per_param_group" / group_params_str
        group_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter DataFrame for the current parameter group
        query = " & ".join([f"`{k}` == {v}" for k, v in group_params_dict.items()])
        group_df = df.query(query)
        
        if group_df.empty:
            logger_analysis.warning(f"No data for parameter group: {group_params_dict}. Skipping plots for this group.")
            continue
        
        group_metrics = {"parameters": group_params_dict}
        
        # a) Distributions for this group (MAE, TotalLenRatio, RelErrorRatio)
        if 'abs_error' in group_df.columns:
            group_metrics['mae_tokens'] = group_df['abs_error'].mean()
            fig_g_mae, ax_g_mae = plt.subplots(figsize=(8,5))
            sns.histplot(group_df['abs_error'], bins=30, kde=True, ax=ax_g_mae)
            save_plot(fig_g_mae, f"MAE Distribution ({group_params_str})", group_output_dir, f"mae_dist.png", is_subfolder=True)
        
        if 'error_ratio' in group_df.columns: # This is Total Output Length Prediction Ratio
            group_metrics['mean_total_len_pred_ratio'] = group_df['error_ratio'].mean()
            fig_g_tlpr, ax_g_tlpr = plt.subplots(figsize=(8,5))
            sns.histplot(group_df['error_ratio'].dropna().clip(0,3), bins=30, kde=True, ax=ax_g_tlpr)
            ax_g_tlpr.axvline(1.0, color='red', linestyle='--')
            save_plot(fig_g_tlpr, f"Total Len Pred Ratio Dist ({group_params_str})", group_output_dir, f"total_len_pred_ratio_dist.png", is_subfolder=True)
        
        # Use df_rel_err (which has ActualRestLen > 2 filter) for relative error ratio dist.
        # Need to filter df_rel_err for the current group_params_dict
        group_df_rel_err = df_rel_err.query(query) if 'rel_error_abs' in df_rel_err else pd.DataFrame()
        if not group_df_rel_err.empty:
            group_metrics['mean_rel_error_abs_filtered'] = group_df_rel_err['rel_error_abs'].mean()
            fig_g_rer, ax_g_rer = plt.subplots(figsize=(8,5))
            sns.histplot(group_df_rel_err['rel_error_abs'].dropna().clip(0,2), bins=30, kde=True, ax=ax_g_rer)
            save_plot(fig_g_rer, f"Abs Rel Error Ratio Dist (ActualRestLen > 2, {group_params_str})", group_output_dir, f"abs_rel_error_ratio_dist.png", is_subfolder=True)
        
        # b) Tail Error Ratio for this group
        group_df_tail = df_tail.query(query) if 'tail_error_ratio_abs' in df_tail else pd.DataFrame()
        if not group_df_tail.empty:
            group_metrics['mean_tail_abs_error_ratio'] = group_df_tail['tail_error_ratio_abs'].mean(skipna=True)
            # Could also plot its distribution for this group if desired
        
        # c) RMSE for this group
        if 'prediction_error' in group_df.columns:
            group_metrics['rmse_tokens'] = np.sqrt((group_df['prediction_error']**2).mean())
        
        # d) R-squared for this group
        if 'actual_rest_len' in group_df.columns and 'predicted_rest_len' in group_df.columns:
            group_df_r2 = group_df.dropna(subset=['actual_rest_len', 'predicted_rest_len'])
            if len(group_df_r2) > 1:
                group_metrics['r2_score_remaining_len'] = r2_score(group_df_r2['actual_rest_len'], group_df_r2['predicted_rest_len'])
        
        metadata_store['performance_per_param_group'].append(group_metrics)
    
    # --- 3. Trend plots (Error/Ratio vs ActualRestLen/StepIndex) FOR EACH PARAM GROUP ---
    if all(c in df.columns for c in ['actual_rest_len', 'error_ratio', 'prediction_error', 'step_index']):
        for i, group_params_dict in enumerate(tqdm(unique_param_groups, desc="Generating Trend Plots per Group")):
            group_params_str = "_".join([f"{k.replace('_','')[0:3]}{v}" for k, v in group_params_dict.items()]) # Shorter filename
            group_output_dir_trends = base_output_dir / "trends_per_param_group" / group_params_str
            
            query = " & ".join([f"`{k}` == {v}" for k, v in group_params_dict.items()])
            group_df = df.query(query).copy() # Use a copy for adding binned columns
            
            if group_df.empty: continue
            
            # a) Error Ratio vs Actual Rest Length for this group
            max_arl_g = group_df['actual_rest_len'].max()
            min_arl_g = group_df['actual_rest_len'].min()
            if pd.notna(max_arl_g) and pd.notna(min_arl_g) and max_arl_g > min_arl_g:
                num_bins_g = min(10, int((max_arl_g - min_arl_g) / 20) + 1); 
                if num_bins_g < 2: num_bins_g = max(1, int(max_arl_g - min_arl_g)+1) # Ensure some bins
                
                try:
                    group_df['arl_bins_cat'] = pd.cut(group_df['actual_rest_len'], bins=num_bins_g, right=False, include_lowest=True, duplicates='drop')
                    plot_data = group_df.groupby('arl_bins_cat')['error_ratio'].mean().reset_index()
                    plot_data['arl_bins_mid'] = plot_data['arl_bins_cat'].apply(lambda x: x.mid)
                    
                    fig_trend1, ax_trend1 = plt.subplots(figsize=(10,6))
                    sns.lineplot(data=plot_data, x='arl_bins_mid', y='error_ratio', marker='o', ax=ax_trend1)
                    ax_trend1.axhline(1.0, color='red', linestyle='--')
                    ax_trend1.set_xlabel("Actual Remaining Length (Bin Midpoint)")
                    ax_trend1.set_ylabel("Mean Error Ratio (PredTotal/ActualTotal)")
                    save_plot(fig_trend1, f"Error Ratio vs Actual Rest Len ({group_params_str})", group_output_dir_trends, "er_vs_arl.png", is_subfolder=True)
                except Exception as e_plot1:
                    logger_analysis.warning(f"Skipping ER vs ARL plot for group {group_params_str} due to: {e_plot1}")
                    
            
            # b) Error Ratio vs Step Index for this group
            # (Similar logic: bin 'step_index', groupby, plot)
    
    # --- 4. TRAIL-like Heatmap FOR EACH PARAM GROUP ---
    if all(c in df.columns for c in ['actual_rest_len', 'predicted_rest_len']):
        for i, group_params_dict in enumerate(tqdm(unique_param_groups, desc="Generating Heatmaps per Group")):
            group_params_str = "_".join([f"{k.replace('_','')[0:3]}{v}" for k, v in group_params_dict.items()])
            group_output_dir_hm = base_output_dir / "heatmaps_per_param_group" 
            
            query = " & ".join([f"`{k}` == {v}" for k, v in group_params_dict.items()])
            group_df = df.query(query)
            if group_df.empty: continue
            
            plot_length_prediction_heatmap(
                group_df, output_dir=group_output_dir_hm,
                filename=f"consistency_heatmap", # param_group_str will be added by plot_length_prediction_heatmap
                title=f"Consistency Plot", # param_group_str will be added
                max_length_for_binning=group_df['max_new_tok_session'].iloc[0] if not group_df.empty else MAX_MAX_NEW_TOKENS or 512, # Use session max for this group
                num_bins=10,
                param_group_str=group_params_str # Pass for unique filename/title
            )
            
    # --- 5. Scatter plot of Predicted vs Actual (Overall) ---
    # (This was already good, keep it in the "overall" section or here)
    if 'predicted_rest_len' in df.columns and 'actual_rest_len' in df.columns:
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
        sample_df_for_scatter = df.sample(n=min(5000, len(df))) if len(df) > 5000 else df
        sns.scatterplot(data=sample_df_for_scatter, x='actual_rest_len', y='predicted_rest_len',
                        hue='prediction_error', size='latency_ms', palette='coolwarm', alpha=0.6,
                        sizes=(20, 200), ax=ax_scatter)
        # ... (rest of scatter plot code)
        save_plot(fig_scatter, "Overall: Predicted vs Actual Remaining Length (Sampled)", overall_dir, "overall_predicted_vs_actual_scatter.png")
    
    # --- 6. Per-Prompt Plotting (Sampled) ---
    sampled_prompt_ids = df['prompt_id'].drop_duplicates().sample(min(5, df['prompt_id'].nunique())).tolist() # Sample 5 prompts
    prompt_plot_dir = base_output_dir / "per_prompt_ratio_evolution"
    
    if all(col in df.columns for col in ['prompt_id', 'step_index', 'error_ratio'] + param_cols):
        for p_id in tqdm(sampled_prompt_ids, desc="Plotting per-prompt ratio evolution"):
            prompt_df = df[df['prompt_id'] == p_id].copy()
            if prompt_df.empty: continue
            
            # Create a unique identifier for each param group for legend/hue
            prompt_df['param_group_legend'] = prompt_df.apply(
                lambda row: f"T{row['temp']}_K{row['top_k']}_RP{row['rep_p']}_MNT{row['max_new_tok_session']}", axis=1
            )
            
            fig_prompt, ax_prompt = plt.subplots(figsize=(12,7))
            sns.lineplot(data=prompt_df, x='step_index', y='error_ratio', hue='param_group_legend', marker='.', ax=ax_prompt)
            ax_prompt.axhline(1.0, color='red', linestyle='--', label='Ideal Ratio (1.0)')
            ax_prompt.set_xlabel("Decoding Step Index")
            ax_prompt.set_ylabel("Total Output Length Prediction Ratio")
            ax_prompt.legend(title="Decoding Parameters", bbox_to_anchor=(1.05, 1), loc='upper left')
            # Limit y-axis for readability if ratios are extreme
            median_ratio = prompt_df['error_ratio'].median()
            if pd.notna(median_ratio):
                ax_prompt.set_ylim(max(0, median_ratio - 2), median_ratio + 2) # Example: median +/- 2
            else:
                ax_prompt.set_ylim(0,3)
            
            safe_prompt_id_fn = p_id.replace('/','_').replace(':','_') # Make filename safe
            save_plot(fig_prompt, f"Total Length Pred Ratio vs Step for Prompt: {p_id}", prompt_plot_dir, f"prompt_{safe_prompt_id_fn}_ratio_vs_step.png", is_subfolder=True)
    
    logger_analysis.info(f"Analysis complete. Figures and metadata saved to {base_output_dir}")
    # Save metadata_store
    try:
        with open(METADATA_OUTPUT_FILE, 'w') as f_meta:
            json.dump(metadata_store, f_meta, indent=2, cls=NpEncoder)
        logger_analysis.info(f"Aggregated metadata and stats saved to {METADATA_OUTPUT_FILE}")
    except Exception as e_meta:
        logger_analysis.error(f"Could not save metadata file: {e_meta}")


if __name__ == '__main__':
    if not RESULTS_JSONL_FILE.exists() or RESULTS_JSONL_FILE.stat().st_size == 0:
        logger_analysis.critical(f"Results file for analysis not found or is empty: {RESULTS_JSONL_FILE}")
    else:
        results_df = load_results_to_dataframe(RESULTS_JSONL_FILE)
        if not results_df.empty:
            metadata_and_stats_summary = {} # Initialize dict to store results
            analyze_and_plot_results(results_df, FIGURES_OUTPUT_DIR, metadata_and_stats_summary)
        else:
            logger_analysis.error("DataFrame is empty after loading. No analysis performed.")