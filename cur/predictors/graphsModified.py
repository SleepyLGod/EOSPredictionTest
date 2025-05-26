import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any # For type hinting

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger_analysis = logging.getLogger("analysis_script")

# --- Configurations ---
# !! MODIFY these to your actual paths and desired settings !!
RESULTS_JSONL_FILE = Path("length_predictor_eval_results_Meta_Llama_3_70B_20250521_142634.jsonl")
FIGURES_OUTPUT_DIR = Path("./results/detailed_analysis/") 
METADATA_OUTPUT_FILE = FIGURES_OUTPUT_DIR / "aggregated_metadata_and_stats.json"

FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These MIN/MAX constants MUST match those used for normalization when training the length predictor
# and cover the ranges in your DECODING_PARAMS_LIST used in the evaluation script.
MIN_TEMP, MAX_TEMP = 0.1, 0.9
MIN_TOP_K, MAX_TOP_K = 1, 100
MIN_REP_PENALTY, MAX_REP_PENALTY = 1.0, 1.6 # Adjusted based on previous discussions
MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS = 100, 500 # Adjusted based on previous discussions
MIN_SEQ_POS, MAX_SEQ_POS = 0, 8191 # Should match training normalization range for seq_pos


def load_results_to_dataframe(jsonl_file_path: Path) -> pd.DataFrame:
    if not jsonl_file_path.exists():
        logger_analysis.error(f"Results file not found: {jsonl_file_path}")
        return pd.DataFrame()
    
    all_steps_data: List[Dict[str, Any]] = [] # For type hinting
    with open(jsonl_file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            try:
                session_result = json.loads(line)
                prompt_id = session_result.get("prompt_id")
                dec_params = session_result.get("decoding_params", {})
                
                for step_detail in session_result.get("step_predictions", []):
                    # Flatten the structure: bring session-level info to each step
                    flat_step = {
                        "prompt_id": prompt_id,
                        "temp": dec_params.get("temperature"),
                        "top_k": dec_params.get("top_k"),
                        "rep_p": dec_params.get("repetition_penalty"),
                        "max_new_tok_session": dec_params.get("max_new_tokens"), # Max tokens for this session
                        "actual_generated_steps_session": session_result.get("actual_generated_steps"),
                        "eos_encountered_session": session_result.get("eos_encountered_in_session"),
                        **step_detail # Unpack all keys from the step_detail dictionary
                    }
                    all_steps_data.append(flat_step)
            except json.JSONDecodeError:
                logger_analysis.warning(f"Skipping invalid JSON line {line_idx+1} in {jsonl_file_path}")
            except Exception as e:
                logger_analysis.error(f"Error processing line {line_idx+1}: {e}")
    
    if not all_steps_data:
        logger_analysis.warning("No data loaded from results file.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_steps_data)
    logger_analysis.info(f"Loaded {len(df)} step prediction records into DataFrame from {len(set(df['prompt_id'])) if 'prompt_id' in df else 0} unique prompt sessions.")
    
    numeric_cols = [
        'predicted_rest_len', 'actual_rest_len', 'latency_ms', 'prediction_error', 
        'current_full_sequence_len_for_pred', 'step_index', 
        'temp', 'top_k', 'rep_p', 'max_new_tok_session', 'actual_generated_steps_session'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce non-numeric to NaN
    
    # Calculate new metrics: Error Ratio
    if 'current_full_sequence_len_for_pred' in df.columns and \
        'predicted_rest_len' in df.columns and 'actual_rest_len' in df.columns:
        
        df['predicted_total_len_from_step'] = df['current_full_sequence_len_for_pred'] + df['predicted_rest_len']
        df['actual_total_len_from_step'] = df['current_full_sequence_len_for_pred'] + df['actual_rest_len']
        
        # Avoid division by zero for error_ratio; replace 0 or very small actual_total_len with NaN or a small epsilon
        df['error_ratio'] = df['predicted_total_len_from_step'] / df['actual_total_len_from_step'].replace(0, np.nan)
        # df['error_ratio_log'] = np.log(df['error_ratio'].replace(0, np.nan)) # Log ratio can be useful

    # Drop rows where essential numeric conversions failed or key metrics are NaN
    # Add 'error_ratio' to essential columns if it's calculated
    essential_cols_for_analysis = ['prediction_error', 'actual_rest_len', 'latency_ms', 
                                    'temp', 'top_k', 'rep_p', 'max_new_tok_session']
    if 'error_ratio' in df.columns:
        essential_cols_for_analysis.append('error_ratio')
        
    df.dropna(subset=essential_cols_for_analysis, inplace=True)
    logger_analysis.info(f"DataFrame shape after numeric conversion and NaN drop on essential columns: {df.shape}")
    return df

def save_plot(plt_figure, fig_title: str, output_dir: Path, plot_name: str): # Pass figure explicitly
    """Helper to save plots and close them."""
    plt_figure.suptitle(fig_title, fontsize=16)
    plt_figure.tight_layout(rect=[0, 0, 1, 0.96]) 
    full_path = output_dir / plot_name
    plt_figure.savefig(full_path)
    plt.close(plt_figure) # Close the passed figure
    logger_analysis.info(f"Generated and saved: {full_path}")


def re_normalize_for_heatmap_axes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-calculates normalized seq_pos and max_len_session for heatmap axes.
    This is needed if these specific normalized features were not saved in the JSONL
    and you want to plot against them.
    """
    logger_analysis.info("Attempting to re-normalize 'seq_pos' and 'max_len_session' for heatmap axes...")
    
    def _normalize_val(v, min_v, max_v):
        if pd.isna(v): return np.nan
        if max_v == min_v: return 0.0 if v == min_v else 0.5
        norm_v = (v - min_v) / (max_v - min_v)
        return np.clip(norm_v, 0.0, 1.0)

    if 'current_full_sequence_len_for_pred' in df.columns:
        df['norm_seq_pos_axis'] = df['current_full_sequence_len_for_pred'].apply(
            lambda x: _normalize_val(x, MIN_SEQ_POS, MAX_SEQ_POS)
        )
    else:
        logger_analysis.warning("Column 'current_full_sequence_len_for_pred' not found for 'norm_seq_pos_axis'.")
        df['norm_seq_pos_axis'] = np.nan

    if 'max_new_tok_session' in df.columns:
        df['norm_max_len_session_axis'] = df['max_new_tok_session'].apply(
            lambda x: _normalize_val(x, MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS)
        )
    else:
        logger_analysis.warning("Column 'max_new_tok_session' not found for 'norm_max_len_session_axis'.")
        df['norm_max_len_session_axis'] = np.nan
    return df

# --- (plot_performance_heatmap and plot_length_prediction_heatmap from previous response can be inserted here)
# --- Make sure they use the passed plt_figure for saving, or call save_plot correctly.

def analyze_and_plot_results(df: pd.DataFrame, output_dir: Path, metadata_store: dict):
    if df.empty:
        logger_analysis.info("DataFrame is empty, no analysis to perform.")
        return

    # Create 'abs_error' if not present
    if 'prediction_error' in df.columns and 'abs_error' not in df.columns:
        df['abs_error'] = df['prediction_error'].abs()

    # --- 1. Overall Performance Metrics ---
    logger_analysis.info("\n--- Overall Performance Metrics ---")
    overall_metrics = {}
    if 'prediction_error' in df:
        overall_metrics['mae'] = df['prediction_error'].abs().mean()
        overall_metrics['rmse'] = np.sqrt((df['prediction_error']**2).mean())
        overall_metrics['bias'] = df['prediction_error'].mean()
        logger_analysis.info(f"Overall MAE: {overall_metrics['mae']:.2f} tokens")
        logger_analysis.info(f"Overall RMSE: {overall_metrics['rmse']:.2f} tokens")
        logger_analysis.info(f"Overall Bias (Pred - Actual): {overall_metrics['bias']:.2f} tokens")
    
    if 'error_ratio' in df:
        overall_metrics['mean_error_ratio'] = df['error_ratio'].mean()
        overall_metrics['median_error_ratio'] = df['error_ratio'].median()
        logger_analysis.info(f"Overall Mean Error Ratio (PredTotal/ActualTotal): {overall_metrics['mean_error_ratio']:.3f}")
        logger_analysis.info(f"Overall Median Error Ratio: {overall_metrics['median_error_ratio']:.3f}")

    if 'latency_ms' in df:
        overall_metrics['mean_latency_ms'] = df['latency_ms'].mean()
        overall_metrics['median_latency_ms'] = df['latency_ms'].median()
        overall_metrics['p90_latency_ms'] = df['latency_ms'].quantile(0.90)
        logger_analysis.info(f"Overall Mean Latency: {overall_metrics['mean_latency_ms']:.2f} ms")
        # ... (log other latency stats)

    metadata_store['overall_performance'] = overall_metrics
    
    # Plot overall error distributions
    if 'prediction_error' in df:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(df['prediction_error'], bins=50, kde=True, ax=ax1)
        save_plot(fig1, "Distribution of Prediction Errors (All Steps)", output_dir, "prediction_error_distribution.png")
    
    if 'error_ratio' in df:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Filter out extreme ratios for better visualization if necessary, e.g., ratios between 0.1 and 10
        sns.histplot(df['error_ratio'].dropna().clip(0, 5), bins=50, kde=True, ax=ax2) # Clip for viz
        save_plot(fig2, "Distribution of Error Ratios (PredTotal/ActualTotal, Clipped for Viz)", output_dir, "error_ratio_distribution.png")

    # --- 2. Performance by System Parameter Group ---
    param_cols_for_grouping = ['temp', 'top_k', 'rep_p', 'max_new_tok_session']
    grouped_stats = {}

    if all(col in df.columns for col in param_cols_for_grouping + ['abs_error']):
        mae_per_group_df = df.groupby(param_cols_for_grouping)['abs_error'].mean().reset_index().sort_values(by='abs_error')
        logger_analysis.info("\n--- MAE per System Parameter Group (Top & Bottom) ---")
        logger_analysis.info("Top 5 Best (Lowest MAE):\n" + mae_per_group_df.head(5).to_string())
        logger_analysis.info("\nTop 5 Worst (Highest MAE):\n" + mae_per_group_df.tail(5).to_string())
        grouped_stats['mae_per_param_group'] = mae_per_group_df.to_dict(orient='records')
        # ... (MAE barh plots as before, pass the figure to save_plot: fig, axes = plt.subplots(...); save_plot(fig, ...))

    if all(col in df.columns for col in param_cols_for_grouping + ['error_ratio']):
        error_ratio_per_group_df = df.groupby(param_cols_for_grouping)['error_ratio'].mean().reset_index().sort_values(by='error_ratio')
        logger_analysis.info("\n--- Mean Error Ratio per System Parameter Group (Closest to 1 are best) ---")
        # Find those closest to 1
        error_ratio_per_group_df['abs_diff_from_1'] = (error_ratio_per_group_df['error_ratio'] - 1).abs()
        error_ratio_per_group_df = error_ratio_per_group_df.sort_values(by='abs_diff_from_1')
        logger_analysis.info("Top 5 Closest to Ratio=1:\n" + error_ratio_per_group_df.head(5).to_string())
        grouped_stats['mean_error_ratio_per_param_group'] = error_ratio_per_group_df.to_dict(orient='records')


    # Latency violin plot (as before)
    if all(col in df.columns for col in ['temp', 'top_k', 'latency_ms']):
        fig_lat, ax_lat = plt.subplots(figsize=(14, 8))
        df_latency_plot = df.copy()
        df_latency_plot['temp_cat'] = df_latency_plot['temp'].astype(str)
        df_latency_plot['top_k_cat'] = df_latency_plot['top_k'].astype(str)
        sns.violinplot(data=df_latency_plot, x='temp_cat', y='latency_ms', hue='top_k_cat', inner='quartile', cut=0, ax=ax_lat)
        ax_lat.set_ylabel("Latency (ms)")
        ax_lat.set_xlabel("Temperature")
        ax_lat.set_yscale('log') 
        ax_lat.legend(title="Top_K")
        save_plot(fig_lat, "Length Predictor Latency Distribution by Temp & Top_K", output_dir, "latency_violin_by_temp_topk.png")

    metadata_store['performance_by_parameter_group'] = grouped_stats

    # --- 3. Detailed Trend Analysis with FacetGrids ---
    df_trend_analysis = df.copy() # Use a copy for adding binned columns

    # a) Error Ratio vs. Actual Rest Length (Faceted)
    if all(col in df_trend_analysis.columns for col in ['actual_rest_len', 'error_ratio', 'temp', 'top_k']):
        # ... (Binning logic for 'actual_rest_len' into 'arl_bins' and creating 'arl_bin_midpoint' as in previous response)
        # ... (Then FacetGrid plot using 'error_ratio' as y-axis)
        # ... Example:
        # error_ratio_by_bin_temp_topk = df_trend_analysis.groupby(['arl_bins_midpoint', 'temp', 'top_k'])['error_ratio'].mean().reset_index()
        # g = sns.FacetGrid(...)
        # g.map_dataframe(sns.lineplot, x='arl_bins_midpoint', y='error_ratio', marker='o')
        # g.map(plt.axhline, y=1, color='grey', linestyle='--') # Target ratio is 1
        # save_plot(g.figure, "Error Ratio vs Actual Rest Len (Faceted by Temp & TopK)", output_dir, "error_ratio_vs_arl_faceted.png")
        pass # Placeholder for brevity, implement as per previous FacetGrid example, using 'error_ratio'

    # b) Error Ratio vs. Step Index (Faceted)
    if all(col in df_trend_analysis.columns for col in ['step_index', 'error_ratio', 'temp', 'top_k']):
        # Create bins for step_index if it has a large range
        # df_trend_analysis['step_index_binned'] = pd.cut(...)
        # error_ratio_by_step_temp_topk = df_trend_analysis.groupby(['step_index_binned_midpoint', 'temp', 'top_k'])['error_ratio'].mean().reset_index()
        # g = sns.FacetGrid(...)
        # g.map_dataframe(sns.lineplot, x='step_index_binned_midpoint', y='error_ratio', marker='o')
        # g.map(plt.axhline, y=1, color='grey', linestyle='--')
        # save_plot(g.figure, "Error Ratio vs Step Index (Faceted by Temp & TopK)", output_dir, "error_ratio_vs_step_faceted.png")
        pass # Placeholder

    # --- 4. Heatmap Analysis (Inspired by TRAIL paper) ---
    logger_analysis.info("\n--- Generating Heatmaps ---")
    df_for_heatmap = re_normalize_for_heatmap_axes(df.copy()) # Create normalized axes

    # a) Consistency Heatmap (Frequency based, similar to TRAIL Figure 4)
    # (plot_length_prediction_heatmap function from previous response would be called here)
    # plot_length_prediction_heatmap(df, output_dir, "heatmap_consistency_overall", "Overall Consistency: Pred vs GT Bins", max_length_for_binning=MAX_MAX_NEW_TOKENS or 512)
    pass # Placeholder

    # b) Performance Heatmap (MAE or Error Ratio vs. Normalized Features)
    if 'norm_seq_pos_axis' in df_for_heatmap.columns and 'norm_max_len_session_axis' in df_for_heatmap.columns and 'abs_error' in df_for_heatmap.columns:
        # Bin the normalized features (0-1 range typically)
        df_for_heatmap['norm_seq_pos_bin'] = pd.cut(df_for_heatmap['norm_seq_pos_axis'], bins=np.linspace(0,1,11), labels=[f"{i*0.1:.1f}-{(i+1)*0.1:.1f}" for i in range(10)], include_lowest=True, right=True)
        df_for_heatmap['norm_max_len_bin'] = pd.cut(df_for_heatmap['norm_max_len_session_axis'], bins=np.linspace(0,1,6), labels=[f"{i*0.2:.1f}-{(i+1)*0.2:.1f}" for i in range(5)], include_lowest=True, right=True)
        
        # Heatmap for MAE
        # mae_heatmap_data = pd.pivot_table(df_for_heatmap, values='abs_error', index='norm_max_len_bin', columns='norm_seq_pos_bin', aggfunc='mean')
        # fig_h_mae, ax_h_mae = plt.subplots(figsize=(12,8))
        # sns.heatmap(mae_heatmap_data, annot=True, fmt=".2f", cmap="viridis_r", linewidths=.5, ax=ax_h_mae)
        # save_plot(fig_h_mae, "MAE Heatmap (Norm. MaxLen vs Norm. SeqPos)", output_dir, "heatmap_mae_vs_norm_features.png")
        pass # Placeholder

        # Heatmap for Error Ratio
        # error_ratio_heatmap_data = pd.pivot_table(df_for_heatmap, values='error_ratio', index='norm_max_len_bin', columns='norm_seq_pos_bin', aggfunc='mean')
        # fig_h_er, ax_h_er = plt.subplots(figsize=(12,8))
        # sns.heatmap(error_ratio_heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", center=1.0, linewidths=.5, ax=ax_h_er)
        # save_plot(fig_h_er, "Error Ratio Heatmap (Norm. MaxLen vs Norm. SeqPos)", output_dir, "heatmap_error_ratio_vs_norm_features.png")
        pass # Placeholder

    # --- 5. Scatter Plot of Predicted vs Actual (as before) ---
    # ... (Scatter plot code, pass figure to save_plot)

    logger_analysis.info(f"Analysis complete. Figures and metadata saved to {output_dir}")
    # Save metadata_store
    try:
        with open(METADATA_OUTPUT_FILE, 'w') as f_meta:
            json.dump(metadata_store, f_meta, indent=2, cls=NpEncoder) # Use a NumpyEncoder if metadata_store contains numpy types
        logger_analysis.info(f"Aggregated metadata and stats saved to {METADATA_OUTPUT_FILE}")
    except Exception as e_meta:
        logger_analysis.error(f"Could not save metadata file: {e_meta}")


# Helper NpEncoder for JSON serialization if metadata contains numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    if not RESULTS_JSONL_FILE.exists() or RESULTS_JSONL_FILE.stat().st_size == 0:
        logger_analysis.error(f"Results file for analysis not found or is empty: {RESULTS_JSONL_FILE}")
    else:
        results_df = load_results_to_dataframe(RESULTS_JSONL_FILE)
        if not results_df.empty:
            # Initialize a dictionary to store aggregated metadata and stats
            metadata_and_stats_summary = {}
            analyze_and_plot_results(results_df, FIGURES_OUTPUT_DIR, metadata_and_stats_summary)
        else:
            logger_analysis.error("Failed to load data or data is empty after loading. No analysis performed.")