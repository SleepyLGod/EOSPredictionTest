import random
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
RESULTS_JSONL_FILE_LIST = [
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250527_194051.jsonl"), # Databricks
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250526_151530.jsonl"), # Clean
    Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250529_000246.jsonl"), # Eval
]
FIGURES_OUTPUT_DIR_LIST = [
    Path("./results/databricks/"),
    Path("./results/clean/"),
    Path("./results/eval/"),
]

RESULTS_JSONL_FILE = RESULTS_JSONL_FILE_LIST[0]
FIGURES_OUTPUT_DIR = FIGURES_OUTPUT_DIR_LIST[0]
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
NUM_SAMPLED_PROMPTS_FOR_EVOLUTION = 10
MAX_LEGEND_ITEMS_PER_PLOT = 10

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, pd.Interval): return str(obj) # Handle pandas Interval for bin labels
        return super(NpEncoder, self).default(obj)

# --- Function to load and preprocess data ---
def load_results_to_dataframe(jsonl_file_path: Path) -> pd.DataFrame:
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


# --- Helper function to save plots ---
def save_plot(plt_figure, main_title: str, output_dir: Path, plot_name: str, is_facetgrid: bool = False):
    """Helper to save plots and close them, supports subfolders and FacetGrid title handling."""
    if not is_facetgrid: # FacetGrids handle their own main titles (suptitle) often
        plt_figure.suptitle(main_title, fontsize=14, y=0.98) # Reduced suptitle font size, adjusted y
    else: # For FacetGrids, we might set individual subplot titles or a general title before calling this
        # If FacetGrid's main title needs setting, it's usually done via g.fig.suptitle()
        pass # Assume FacetGrid titles are handled by the FacetGrid object itself

    plt_figure.tight_layout(rect=[0, 0, 1, 0.95 if not is_facetgrid else 1]) 
    
    final_path = output_dir / plot_name
    final_path.parent.mkdir(parents=True, exist_ok=True) 
    
    plt_figure.savefig(final_path)
    plt.close(plt_figure)
    logger_analysis.info(f"Generated and saved: {final_path}")


# --- Function for TRAIL-like heatmap ---
def plot_length_prediction_heatmap(
    df: pd.DataFrame, 
    output_dir: Path, 
    filename_prefix: str,
    base_title: str,
    max_length_for_binning: int = 512, 
    num_bins: int = 10,
    param_group_str: Optional[str] = None # For filename/title if per group
):
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
    
    fig, ax = plt.subplots(figsize=(10, 8)) # Create figure and axes
    sns.heatmap(log_contingency_table, annot=False, fmt=".1f", cmap="Blues", linewidths=.5, cbar_kws={'label': 'Log(Count + 1)'}, ax=ax)
    
    plot_title_text = f"{base_title} ({param_group_str})" if param_group_str else base_title
    ax.set_title(plot_title_text, fontsize=12) # Set title on axes, reduced fontsize
    ax.set_xlabel("Groundtruth Remaining Length (Binned)")
    ax.set_ylabel("Predicted Remaining Length (Binned)")
    
    final_plot_filename = f"{filename_prefix}{'_'+param_group_str if param_group_str else ''}.png"
    save_plot(fig, "", output_dir, final_plot_filename, is_facetgrid=False) # Main title handled by ax.set_title


# --- Main Analysis and Plotting Function ---
def analyze_and_plot_results(df: pd.DataFrame, base_output_dir: Path, metadata_store: dict):
    if df.empty:
        logger_analysis.info("DataFrame is empty, no analysis to perform.")
        return
    
    param_cols = ['temp', 'top_k', 'rep_p', 'max_new_tok_session'] # Define parameter columns for grouping
    
    # --- 1. Overall Performance Metrics & Distributions (All Prompts, All Params) ---
    overall_dir = base_output_dir / "overall"
    overall_dir.mkdir(parents=True, exist_ok=True)
    logger_analysis.info("\n--- Overall Performance Metrics (All Data) ---")
    
    overall_metrics = {}
    if 'prediction_error' in df.columns:
        # df['abs_error'] should be pre-calculated in load_results_to_dataframe
        if 'abs_error' not in df.columns: df['abs_error'] = df['prediction_error'].abs()
        overall_metrics['mae_tokens'] = df['abs_error'].mean()
        overall_metrics['rmse_tokens'] = np.sqrt((df['prediction_error']**2).mean())
        overall_metrics['bias_tokens'] = df['prediction_error'].mean()
        
        fig_err_dist, ax_err_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df['prediction_error'], bins=50, kde=True, ax=ax_err_dist)
        ax_err_dist.set_title("Prediction Error (tokens)", fontsize=12) # Set axes title
        save_plot(fig_err_dist, "Overall: Distribution of Prediction Error (tokens)", overall_dir, "overall_prediction_error_dist.png")
    
    if 'error_ratio' in df.columns: # This is Total Output Length Prediction Ratio
        overall_metrics['mean_total_len_pred_ratio'] = df['error_ratio'].mean()
        overall_metrics['median_total_len_pred_ratio'] = df['error_ratio'].median()
        
        fig_er_dist, ax_er_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df['error_ratio'].dropna().clip(0, 3), bins=50, kde=True, ax=ax_er_dist)
        ax_er_dist.axvline(1.0, color='red', linestyle='--', label='Ideal Ratio (1.0)')
        ax_er_dist.legend()
        ax_er_dist.set_title("Total Output Length Prediction Ratio (PredTotal/ActualTotal, Clipped)", fontsize=12)
        save_plot(fig_er_dist, "Overall: Distribution of Total Output Length Prediction Ratio", overall_dir, "overall_total_len_pred_ratio_dist.png")
    
    # Relative Error Ratios for Remaining Length (Filtered for actual_rest_len > TAIL_LENGTH_THRESHOLD)
    df_non_tail_rel_err = df[df['actual_rest_len'] > TAIL_LENGTH_THRESHOLD].copy()
    if not df_non_tail_rel_err.empty and 'actual_rest_len' in df_non_tail_rel_err and 'prediction_error' in df_non_tail_rel_err:
        epsilon = 1e-6 
        df_non_tail_rel_err['rel_error_signed_non_tail'] = df_non_tail_rel_err['prediction_error'] / (df_non_tail_rel_err['actual_rest_len'] + epsilon)
        df_non_tail_rel_err['rel_error_abs_non_tail'] = df_non_tail_rel_err['rel_error_signed_non_tail'].abs()
        overall_metrics['mean_rel_error_abs_non_tail'] = df_non_tail_rel_err['rel_error_abs_non_tail'].mean()
        
        fig_rel_err_dist, ax_rel_err_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df_non_tail_rel_err['rel_error_abs_non_tail'].dropna().clip(0, 2), bins=50, kde=True, ax=ax_rel_err_dist)
        ax_rel_err_dist.set_title(f"Absolute Relative Error on Rem. Len. (ActualRestLen > {TAIL_LENGTH_THRESHOLD}, Clipped)", fontsize=12)
        save_plot(fig_rel_err_dist, f"Overall: Dist of Abs Rel Error (ActualRestLen > {TAIL_LENGTH_THRESHOLD})", overall_dir, "overall_abs_rel_error_dist_non_tail.png")
    
    # Tail Error Ratio Analysis
    df_tail = df[df['actual_rest_len'] <= TAIL_LENGTH_THRESHOLD].copy()
    if not df_tail.empty and 'actual_rest_len' in df_tail and 'prediction_error' in df_tail:
        epsilon = 1e-6
        df_tail['tail_error_ratio_signed'] = df_tail['prediction_error'] / (df_tail['actual_rest_len'] + epsilon)
        df_tail['tail_error_ratio_abs'] = df_tail['tail_error_ratio_signed'].abs()
        overall_metrics['mean_tail_abs_error_ratio'] = df_tail['tail_error_ratio_abs'].mean(skipna=True)
        overall_metrics['median_tail_abs_error_ratio'] = df_tail['tail_error_ratio_abs'].median(skipna=True)
        logger_analysis.info(f"Overall Mean Absolute Tail Error Ratio (ActualRestLen <= {TAIL_LENGTH_THRESHOLD}): {overall_metrics.get('mean_tail_abs_error_ratio', 'N/A'):.3f}")
        
        fig_tail_err_dist, ax_tail_err_dist = plt.subplots(figsize=(10,6))
        sns.histplot(df_tail['tail_error_ratio_abs'].dropna().clip(0,5), bins=30, kde=True, ax=ax_tail_err_dist)
        ax_tail_err_dist.set_title(f"Absolute Tail Error Ratio (ActualRestLen <= {TAIL_LENGTH_THRESHOLD}, Clipped)", fontsize=12)
        save_plot(fig_tail_err_dist, f"Overall: Distribution of Absolute Tail Error Ratio (ActualRestLen <= {TAIL_LENGTH_THRESHOLD})", overall_dir, "overall_tail_abs_error_ratio_dist.png")
    
    # R-squared Score (Overall) & Latency (Overall)
    if 'actual_rest_len' in df.columns and 'predicted_rest_len' in df.columns:
        df_r2 = df.dropna(subset=['actual_rest_len', 'predicted_rest_len'])
        if len(df_r2) > 1: 
            overall_metrics['r2_score_remaining_len'] = r2_score(df_r2['actual_rest_len'], df_r2['predicted_rest_len'])
    if 'latency_ms' in df.columns:
        fig_lat_overall, ax_lat_overall = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, y='latency_ms', ax=ax_lat_overall, cut=0)
        ax_lat_overall.set_yscale('log'); ax_lat_overall.set_title("Predictor Latency", fontsize=12)
        save_plot(fig_lat_overall, "Overall Predictor Latency Distribution", overall_dir, "overall_latency_violin.png")
    
    metadata_store['overall_performance'] = overall_metrics
    logger_analysis.info(f"Overall MAE tokens: {overall_metrics.get('mae_tokens', 'N/A'):.2f}")
    
    
    # --- Per Parameter Group Analysis & Plots ---
    logger_analysis.info("\n--- Generating Plots and Metrics per System Parameter Group ---")
    unique_param_groups = df[param_cols].drop_duplicates().to_dict('records')
    metadata_store['performance_per_param_group'] = []
    
    for i, group_params_dict in enumerate(tqdm(unique_param_groups, desc="Analyzing Parameter Groups")):
        group_params_str_for_file = "_".join([f"{k.replace('_','')[0:3]}{v}" for k, v in group_params_dict.items()])
        group_params_title_str = ", ".join([f"{k.replace('_','').replace('temperature','T').replace('tokens','tok')}={v}" for k, v in group_params_dict.items()])
        
        group_output_dir = base_output_dir / "per_param_group" / group_params_str_for_file
        group_output_dir.mkdir(parents=True, exist_ok=True) # Individual folder for each group
        
        query = " & ".join([f"`{k}` == {v}" for k, v in group_params_dict.items()])
        group_df = df.query(query)
        if group_df.empty: continue
        
        group_metrics = {"parameters": group_params_dict}
        
        # a) Distribution plots for this group
        if 'abs_error' in group_df:
            group_metrics['mae_tokens'] = group_df['abs_error'].mean()
            fig_g_mae, ax_g_mae = plt.subplots(figsize=(8,5)); sns.histplot(group_df['abs_error'], bins=30, kde=True, ax=ax_g_mae)
            ax_g_mae.set_title(f"MAE Distribution", fontsize=10)
            save_plot(fig_g_mae, f"MAE Distribution\n({group_params_title_str})", group_output_dir, f"mae_dist.png", is_facetgrid=True)
        
        if 'error_ratio' in group_df:
            group_metrics['mean_total_len_pred_ratio'] = group_df['error_ratio'].mean()
            fig_g_tlpr, ax_g_tlpr = plt.subplots(figsize=(8,5)); sns.histplot(group_df['error_ratio'].dropna().clip(0,3), bins=30, kde=True, ax=ax_g_tlpr)
            ax_g_tlpr.axvline(1.0, color='red', linestyle='--'); ax_g_tlpr.set_title("Total Len Pred Ratio Dist", fontsize=10)
            save_plot(fig_g_tlpr, f"Total Len Pred Ratio Dist\n({group_params_title_str})", group_output_dir, f"total_len_pred_ratio_dist.png", is_facetgrid=True)
        
        # For this group: filter for non-tail for rel_error_abs, and for tail for tail_error_ratio
        group_df_rel_err = group_df[group_df['actual_rest_len'] > TAIL_LENGTH_THRESHOLD].copy()
        if not group_df_rel_err.empty and 'actual_rest_len' in group_df_rel_err and 'prediction_error' in group_df_rel_err:
            epsilon = 1e-6
            group_df_rel_err['rel_error_abs_non_tail'] = (group_df_rel_err['prediction_error'] / (group_df_rel_err['actual_rest_len'] + epsilon)).abs()
            group_metrics['mean_rel_error_abs_filtered'] = group_df_rel_err['rel_error_abs_non_tail'].mean()
            fig_g_rer, ax_g_rer = plt.subplots(figsize=(8,5)); sns.histplot(group_df_rel_err['rel_error_abs_non_tail'].dropna().clip(0,2), bins=30, kde=True, ax=ax_g_rer)
            ax_g_rer.set_title(f"Abs Rel Error Ratio (ActualRestLen > {TAIL_LENGTH_THRESHOLD})", fontsize=10)
            save_plot(fig_g_rer, f"Abs Rel Error Ratio Dist (ActualRestLen > {TAIL_LENGTH_THRESHOLD})\n({group_params_title_str})", group_output_dir, f"abs_rel_error_ratio_dist.png", is_facetgrid=True)
        
        group_df_tail = group_df[group_df['actual_rest_len'] <= TAIL_LENGTH_THRESHOLD].copy()
        if not group_df_tail.empty and 'actual_rest_len' in group_df_tail and 'prediction_error' in group_df_tail:
            epsilon = 1e-6
            group_df_tail['tail_error_ratio_abs'] = (group_df_tail['prediction_error'] / (group_df_tail['actual_rest_len'] + epsilon)).abs()
            group_metrics['mean_tail_abs_error_ratio'] = group_df_tail['tail_error_ratio_abs'].mean(skipna=True)
        
        if 'prediction_error' in group_df: group_metrics['rmse_tokens'] = np.sqrt((group_df['prediction_error']**2).mean())
        if 'actual_rest_len' in group_df and 'predicted_rest_len' in group_df:
            group_df_r2 = group_df.dropna(subset=['actual_rest_len', 'predicted_rest_len'])
            if len(group_df_r2) > 1: group_metrics['r2_score_remaining_len'] = r2_score(group_df_r2['actual_rest_len'], group_df_r2['predicted_rest_len'])
        
        metadata_store['performance_per_param_group'].append(group_metrics)
        
        # b) Trend plots for this group (Error Ratio vs Actual Rest Len)
        if all(c in group_df.columns for c in ['actual_rest_len', 'error_ratio']):
            max_arl_g = group_df['actual_rest_len'].max(); min_arl_g = group_df['actual_rest_len'].min()
            if pd.notna(max_arl_g) and pd.notna(min_arl_g) and max_arl_g > min_arl_g:
                num_bins_g = min(10, int((max_arl_g - min_arl_g) / 20) + 1); 
                if num_bins_g < 2: num_bins_g = max(2, int(group_df['actual_rest_len'].nunique()/2) if group_df['actual_rest_len'].nunique() > 1 else 2 ) 
                try:
                    group_df_trend = group_df.copy() # Work on a copy
                    group_df_trend['arl_bins_cat'] = pd.cut(group_df_trend['actual_rest_len'], bins=num_bins_g, right=False, include_lowest=True, duplicates='drop')
                    plot_data = group_df_trend.groupby('arl_bins_cat')['error_ratio'].mean().reset_index()
                    plot_data['arl_bins_mid'] = plot_data['arl_bins_cat'].apply(lambda x: x.mid if isinstance(x, pd.Interval) else np.nan) # Check if interval
                    plot_data.dropna(subset=['arl_bins_mid'], inplace=True) # Drop if not an interval (e.g. if cut failed to make intervals)
                    
                    if not plot_data.empty:
                        fig_trend1, ax_trend1 = plt.subplots(figsize=(10,6))
                        sns.lineplot(data=plot_data, x='arl_bins_mid', y='error_ratio', marker='o', ax=ax_trend1)
                        ax_trend1.axhline(1.0, color='red', linestyle='--'); ax_trend1.set_xlabel("Actual Remaining Length (Bin Midpoint)")
                        ax_trend1.set_ylabel("Mean Total Length Pred Ratio"); ax_trend1.set_title("Error Ratio vs Actual Rest Len", fontsize=10)
                        save_plot(fig_trend1, f"Error Ratio vs Actual Rest Len\n({group_params_title_str})", group_output_dir, "er_vs_arl.png", is_facetgrid=True)
                except Exception as e_plot1:
                    logger_analysis.warning(f"Skipping ER vs ARL plot for group {group_params_str_for_file} due to: {e_plot1}")
        
        # c) TRAIL-like Heatmap for this group
        if all(c in group_df.columns for c in ['actual_rest_len', 'predicted_rest_len']):
            plot_length_prediction_heatmap(
                group_df, output_dir=group_output_dir,
                filename_prefix="consistency_heatmap", 
                base_title="Consistency Plot", 
                max_length_for_binning=group_df['max_new_tok_session'].iloc[0] if not group_df.empty else MAX_MAX_NEW_TOKENS or 512,
                num_bins=10,
                param_group_str=group_params_str_for_file 
            )
    
    # --- Per-Prompt Plotting (Sampled) ---
    prompt_plot_dir_base = base_output_dir / "per_prompt_evolution" # Base directory for all prompt plots
    
    if 'prompt_id' in df.columns and df['prompt_id'].nunique() > 0:
        unique_prompt_ids = df['prompt_id'].unique()
        num_prompts_to_sample_viz = min(NUM_SAMPLED_PROMPTS_FOR_EVOLUTION, len(unique_prompt_ids))
        sampled_prompt_ids_for_viz = random.sample(list(unique_prompt_ids), num_prompts_to_sample_viz)
        logger_analysis.info(f"\n--- Generating Per-Prompt Evolution Plots for {num_prompts_to_sample_viz} Sampled Prompts ---")
        
        for p_id in tqdm(sampled_prompt_ids_for_viz, desc="Plotting per-prompt ratio evolution"):
            prompt_df_full = df[df['prompt_id'] == p_id].copy()
            if prompt_df_full.empty: continue
            
            # Create a sub-folder for this specific prompt
            safe_prompt_id_fn = p_id.replace('/','_').replace(':','_').replace('[','').replace(']','')[:50] # Shorter & safer filename
            current_prompt_evolution_dir = prompt_plot_dir_base / safe_prompt_id_fn
            current_prompt_evolution_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot 1: Overview with all param groups for this prompt
            prompt_df_full['param_group_legend'] = prompt_df_full.apply(
                lambda row: f"T{row['temp']}_K{row['top_k']}_RP{row['rep_p']}_MNT{row['max_new_tok_session']}", axis=1
            )
            
            fig_prompt_all_params, ax_prompt_all_params = plt.subplots(figsize=(14,8))
            # Select a subset of param groups for legend if too many, or use smaller markers
            unique_legends = prompt_df_full['param_group_legend'].unique()
            if len(unique_legends) > MAX_LEGEND_ITEMS_PER_PLOT:
                logger_analysis.warning(f"Prompt {p_id} has {len(unique_legends)} param groups, legend might be crowded for overview plot.")
                # Could sample legends or plot without hue if too many. For now, plot all.
            
            sns.lineplot(data=prompt_df_full, x='step_index', y='error_ratio', hue='param_group_legend', marker='.', markersize=5, ax=ax_prompt_all_params, legend="brief")
            ax_prompt_all_params.axhline(1.0, color='red', linestyle='--'); ax_prompt_all_params.set_xlabel("Decoding Step Index")
            ax_prompt_all_params.set_ylabel("Total Output Length Prediction Ratio")
            ax_prompt_all_params.set_title(f"Prompt: {p_id[:50]}...", fontsize=10) # Shorter title
            # Adjust legend
            handles, labels = ax_prompt_all_params.get_legend_handles_labels()
            if len(labels) > MAX_LEGEND_ITEMS_PER_PLOT :
                ax_prompt_all_params.legend(handles[:MAX_LEGEND_ITEMS_PER_PLOT], labels[:MAX_LEGEND_ITEMS_PER_PLOT], title="Decoding Params (Sampled)", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
            else:
                ax_prompt_all_params.legend(title="Decoding Parameters", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
            
            save_plot(fig_prompt_all_params, f"Total Length Pred Ratio vs Step\nPrompt: {p_id[:50]}...", current_prompt_evolution_dir, f"overview_all_params.png", is_facetgrid=True)
            
            # Plot 2: Individual plot for each param group for this prompt
            for i_pg_prompt, group_params_dict_prompt in enumerate(prompt_df_full[param_cols].drop_duplicates().to_dict('records')):
                pg_str_prompt = "_".join([f"{k.replace('_','')[0:3]}{v}" for k, v in group_params_dict_prompt.items()])
                query_prompt_pg = " & ".join([f"`{k}` == {v}" for k, v in group_params_dict_prompt.items()])
                single_param_group_prompt_df = prompt_df_full.query(query_prompt_pg)
                
                if single_param_group_prompt_df.empty: continue
                
                fig_prompt_single_param, ax_prompt_single_param = plt.subplots(figsize=(10,6))
                sns.lineplot(data=single_param_group_prompt_df, x='step_index', y='error_ratio', marker='o', ax=ax_prompt_single_param)
                ax_prompt_single_param.axhline(1.0, color='red', linestyle='--')
                ax_prompt_single_param.set_xlabel("Decoding Step Index")
                ax_prompt_single_param.set_ylabel("Total Output Length Prediction Ratio")
                ax_prompt_single_param.set_title(f"Params: {pg_str_prompt}", fontsize=10)
                # Y-axis limit for better comparison if needed
                median_ratio_pg = single_param_group_prompt_df['error_ratio'].median()
                if pd.notna(median_ratio_pg): ax_prompt_single_param.set_ylim(max(0, median_ratio_pg - 1), median_ratio_pg + 1) 
                else: ax_prompt_single_param.set_ylim(0,2)
                
                save_plot(fig_prompt_single_param, f"Total Length Pred Ratio vs Step\nPrompt: {p_id[:50]}...\nParams: {pg_str_prompt}", current_prompt_evolution_dir, f"param_group_{pg_str_prompt}.png", is_facetgrid=True)
    else:
        logger_analysis.info("Skipping per-prompt evolution plots as 'prompt_id' column is missing or no unique prompts.")
    
    
    # --- Scatter Plot (Overall, filtered for non-tail) ---
    df_scatter_non_tail = df[df['actual_rest_len'] > TAIL_LENGTH_THRESHOLD].copy()
    if 'predicted_rest_len' in df_scatter_non_tail.columns and 'actual_rest_len' in df_scatter_non_tail.columns and not df_scatter_non_tail.empty:
        logger_analysis.info(f"\n--- Generating Overall Scatter Plot (ActualRestLen > {TAIL_LENGTH_THRESHOLD}) ---")
        # Sample more points for the scatter plot, e.g., 10000
        num_scatter_samples = min(10000, len(df_scatter_non_tail))
        sample_df_for_scatter = df_scatter_non_tail.sample(n=num_scatter_samples) if len(df_scatter_non_tail) > num_scatter_samples else df_scatter_non_tail
        
        fig_scatter_overall, ax_scatter_overall = plt.subplots(figsize=(12, 10))
        splot = sns.scatterplot(
            data=sample_df_for_scatter, x='actual_rest_len', y='predicted_rest_len',
            hue='prediction_error', size='latency_ms', palette='coolwarm', alpha=0.6,
            sizes=(20, 200), ax=ax_scatter_overall, legend='brief' # 'brief' or False for legend
        )
        max_val_scatter = max(
            sample_df_for_scatter['actual_rest_len'].max(skipna=True), 
            sample_df_for_scatter['predicted_rest_len'].max(skipna=True)
        )
        if pd.notna(max_val_scatter) and max_val_scatter > 0 :
            ax_scatter_overall.plot([0, max_val_scatter], [0, max_val_scatter], color='red', linestyle='--', label='Perfect Prediction (y=x)')
        
        ax_scatter_overall.set_title(f"Predicted vs. Actual Remaining Length (Sampled, ActualRestLen > {TAIL_LENGTH_THRESHOLD})", fontsize=12)
        ax_scatter_overall.set_xlabel("Actual Remaining Length (tokens)")
        ax_scatter_overall.set_ylabel("Predicted Remaining Length (tokens)")
        
        # Improve legend for scatter plot
        handles, labels = ax_scatter_overall.get_legend_handles_labels()
        # Separate hue and size legends if they become too mixed or unreadable
        # For now, try to make it brief or place outside
        if len(handles) > 10 : # Heuristic for too many legend items
            ax_scatter_overall.legend(handles[:5] + handles[-1:], labels[:5] + labels[-1:], title='Error / Latency (Sample)', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            ax_scatter_overall.legend(loc='upper left', fontsize='small')
            
        ax_scatter_overall.grid(True, linestyle='--', alpha=0.5)
        save_plot(fig_scatter_overall, f"Overall: Predicted vs Actual Remaining Length (Sampled, ActualRestLen > {TAIL_LENGTH_THRESHOLD})", 
                    overall_dir, "overall_pred_vs_actual_scatter_non_tail.png", is_facetgrid=True)
    
    
    logger_analysis.info(f"Analysis complete. Figures and metadata saved to {base_output_dir}")
    # Save metadata_store
    try:
        with open(METADATA_OUTPUT_FILE, 'w') as f_meta:
            json.dump(metadata_store, f_meta, indent=2, cls=NpEncoder)
        logger_analysis.info(f"Aggregated metadata and stats saved to {METADATA_OUTPUT_FILE}")
    except Exception as e_meta:
        logger_analysis.error(f"Could not save metadata file: {e_meta}")


def analyze_and_plot_results_scatter_pro(df: pd.DataFrame, base_output_dir: Path, metadata_store: dict):
    if df.empty:
        logger_analysis.info("DataFrame is empty, no analysis to perform.")
        return
    # --- Scatter Plot (Overall, filtered for non-tail) ---
    scatter_plot_dir = base_output_dir / "overall"
    scatter_plot_dir.mkdir(parents=True, exist_ok=True)
    
    num_scatter_samples_ratio = min(10000, len(df))
    df_scatter_sample = df.sample(n=num_scatter_samples_ratio) if len(df) > num_scatter_samples_ratio else df.copy()
    
    # Scatter Plot 1: Error Ratio vs. Actual Remaining Length
    if 'error_ratio' in df_scatter_sample.columns and 'actual_rest_len' in df_scatter_sample.columns:
        logger_analysis.info(f"\n--- Generating Scatter Plot: Error Ratio vs. Actual Rest Len ---")
        fig_sc_er, ax_sc_er = plt.subplots(figsize=(12, 10))
        sns.scatterplot(
            data=df_scatter_sample,
            x='actual_rest_len',
            y='error_ratio',
            hue='temp', # Example: color by temperature
            size='latency_ms',
            palette='viridis', # Choose a suitable palette
            alpha=0.6,
            sizes=(20, 200),
            ax=ax_sc_er,
            legend='brief'
        )
        ax_sc_er.axhline(1.0, color='red', linestyle='--', label='Ideal Ratio (1.0)')
        ax_sc_er.set_title(f"Total Length Prediction Ratio vs. Actual Remaining Length (Sampled)", fontsize=12)
        ax_sc_er.set_xlabel("Actual Remaining Length (tokens)")
        ax_sc_er.set_ylabel("Total Length Prediction Ratio (PredTotal/ActualTotal)")
        ax_sc_er.set_ylim(df_scatter_sample['error_ratio'].quantile(0.01), df_scatter_sample['error_ratio'].quantile(0.99)) # Zoom on main distribution
        ax_sc_er.legend(loc='upper right', fontsize='small')
        ax_sc_er.grid(True, linestyle='--', alpha=0.5)
        save_plot(fig_sc_er, "Scatter: Total Length Pred Ratio vs Actual Rest Len", 
                    scatter_plot_dir, "scatter_error_ratio_vs_actual_rest_len.png", is_facetgrid=True) # is_facetgrid=True to use suptitle
        
        logger_analysis.info(f"Analysis complete. Figures and metadata saved to {base_output_dir}")


if __name__ == '__main__':
    if not RESULTS_JSONL_FILE.exists() or RESULTS_JSONL_FILE.stat().st_size == 0:
        logger_analysis.critical(f"Results file for analysis not found or is empty: {RESULTS_JSONL_FILE}")
    else:
        results_df = load_results_to_dataframe(RESULTS_JSONL_FILE)
        if not results_df.empty:
            metadata_and_stats_summary = {} 
            analyze_and_plot_results(results_df, FIGURES_OUTPUT_DIR, metadata_and_stats_summary)
            analyze_and_plot_results_scatter_pro(results_df, FIGURES_OUTPUT_DIR, metadata_and_stats_summary)
        else:
            logger_analysis.error("DataFrame is empty after loading. No analysis performed.")
