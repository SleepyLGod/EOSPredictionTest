import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger_analysis = logging.getLogger("analysis_script")

# --- Configurations ---
# RESULTS_JSONL_FILE = Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250521_010529.jsonl") 
# RESULTS_JSONL_FILE = Path("length_predictor_eval_results_Meta_Llama_3_70B_20250521_140235.jsonl")
RESULTS_JSONL_FILE = Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250521_142634.jsonl")
FIGURES_OUTPUT_DIR = Path("./results/databricks/")
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS = 300, 500


def load_results_to_dataframe(jsonl_file_path: Path) -> pd.DataFrame:
    if not jsonl_file_path.exists():
        logger_analysis.error(f"Results file not found: {jsonl_file_path}")
        return pd.DataFrame()
    all_steps_data = []
    with open(jsonl_file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            try:
                session_result = json.loads(line)
                prompt_id = session_result.get("prompt_id")
                dec_params = session_result.get("decoding_params", {})
                
                for step_detail in session_result.get("step_predictions", []):
                    flat_step = {
                        "prompt_id": prompt_id,
                        "temp": dec_params.get("temperature"),
                        "top_k": dec_params.get("top_k"),
                        "rep_p": dec_params.get("repetition_penalty"),
                        "max_new_tok_session": dec_params.get("max_new_tokens"),
                        "actual_generated_steps_session": session_result.get("actual_generated_steps"),
                        "eos_encountered_session": session_result.get("eos_encountered_in_session"),
                        **step_detail
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
    logger_analysis.info(f"Loaded {len(df)} step prediction records into DataFrame.")
    
    numeric_cols = [
        'predicted_rest_len', 'actual_rest_len', 'latency_ms', 'prediction_error', 
        'current_full_sequence_len_for_pred', 'step_index', 
        'temp', 'top_k', 'rep_p', 'max_new_tok_session', 'actual_generated_steps_session'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where essential numeric conversions failed for analysis
    df.dropna(subset=['prediction_error', 'actual_rest_len', 'latency_ms', 
                        'temp', 'top_k', 'rep_p', 'max_new_tok_session'], inplace=True)
    logger_analysis.info(f"DataFrame shape after numeric conversion and NaN drop: {df.shape}")
    return df

def save_plot(fig_title: str, output_dir: Path, plot_name: str):
    """Helper to save plots and close them."""
    plt.suptitle(fig_title, fontsize=16) # Add an overall title to the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    full_path = output_dir / plot_name
    plt.savefig(full_path)
    plt.close()
    logger_analysis.info(f"Generated and saved: {full_path}")


def analyze_and_plot_results(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        logger_analysis.info("DataFrame is empty, no analysis to perform.")
        return
    # --- 1. Overall Performance ---
    logger_analysis.info("\n--- Overall Performance Metrics ---")
    overall_mae = df['prediction_error'].abs().mean()
    overall_rmse = np.sqrt((df['prediction_error']**2).mean())
    overall_bias = df['prediction_error'].mean()
    logger_analysis.info(f"Overall MAE: {overall_mae:.2f} tokens")
    logger_analysis.info(f"Overall RMSE: {overall_rmse:.2f} tokens")
    logger_analysis.info(f"Overall Bias (Pred - Actual): {overall_bias:.2f} tokens")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['prediction_error'], bins=50, kde=True)
    save_plot("Distribution of Prediction Errors (All Steps)", output_dir, "prediction_error_distribution.png")
    
    # --- 2. Performance by System Parameter Group ---
    param_cols_for_grouping = ['temp', 'top_k', 'rep_p', 'max_new_tok_session']
    
    # Ensure no NaN in prediction_error before abs() for MAE
    df['abs_error'] = df['prediction_error'].abs()
    
    # MAE for each parameter combination
    mae_per_group = df.groupby(param_cols_for_grouping)['abs_error'].mean().reset_index()
    mae_per_group = mae_per_group.sort_values(by='abs_error')
    logger_analysis.info("\n--- MAE per System Parameter Group (Top 10 Best & Worst) ---")
    logger_analysis.info("Top 10 Best (Lowest MAE):")
    logger_analysis.info(mae_per_group.head(10).to_string())
    logger_analysis.info("\nTop 10 Worst (Highest MAE):")
    logger_analysis.info(mae_per_group.tail(10).to_string())
    
    # Plot MAE for top/bottom N (example with top 15)
    if len(mae_per_group) > 0:
        num_to_plot = min(15, len(mae_per_group))
        fig, axes = plt.subplots(1, 2, figsize=(18, max(8, num_to_plot * 0.5))) # Adjust height
        mae_per_group.head(num_to_plot).set_index(param_cols_for_grouping)['abs_error'].plot(kind='barh', ax=axes[0])
        axes[0].set_title(f'Top {num_to_plot} Lowest MAE by Parameter Group')
        axes[0].set_xlabel("Mean Absolute Error (tokens)")
        
        mae_per_group.tail(num_to_plot).set_index(param_cols_for_grouping)['abs_error'].plot(kind='barh', ax=axes[1])
        axes[1].set_title(f'Top {num_to_plot} Highest MAE by Parameter Group')
        axes[1].set_xlabel("Mean Absolute Error (tokens)")
        save_plot("MAE by Parameter Group (Best & Worst)", output_dir, "mae_param_groups_barh.png")
    # Latency distribution (Violin plot for more detail)
    plt.figure(figsize=(14, 8))
    # Ensure 'temp' and 'top_k' are treated as categories for plotting if they are numeric
    df_latency_plot = df.copy()
    df_latency_plot['temp_cat'] = df_latency_plot['temp'].astype(str) # Treat as category for distinct x-ticks
    df_latency_plot['top_k_cat'] = df_latency_plot['top_k'].astype(str)
    sns.violinplot(data=df_latency_plot, x='temp_cat', y='latency_ms', hue='top_k_cat', inner='quartile', cut=0)
    plt.ylabel("Latency (ms)")
    plt.xlabel("Temperature")
    plt.yscale('log') 
    plt.legend(title="Top_K")
    save_plot("Length Predictor Latency Distribution by Temperature and Top_K", output_dir, "latency_violin_by_temp_topk.png")
    # --- 3. Detailed Contrastive Analysis ---
    # a) Prediction Error vs. Actual Remaining Length, faceted by Temperature & Top_K
    if 'actual_rest_len' in df.columns and 'prediction_error' in df.columns:
        # Ensure 'actual_rest_len' is numeric and drop NaNs before binning
        df_bin_analysis = df.dropna(subset=['actual_rest_len', 'prediction_error', 'temp', 'top_k'])
        
        max_arl = df_bin_analysis['actual_rest_len'].max()
        min_arl = df_bin_analysis['actual_rest_len'].min()
        
        if pd.notna(max_arl) and pd.notna(min_arl) and max_arl > min_arl:
            # Define bin edges, e.g., every 20 or 50 tokens, or using quantiles
            num_bins = min(15, int((max_arl - min_arl) / 20) +1) # Max 10 bins, or bins of size 20
            if num_bins < 2 : num_bins = 5 # Ensure at least a few bins if range is small
            try:
                df_bin_analysis['arl_bins'] = pd.cut(df_bin_analysis['actual_rest_len'], bins=num_bins, right=False, include_lowest=True)
            except ValueError as e_cut: # Handle cases where pd.cut fails (e.g. not enough unique values)
                logger_analysis.warning(f"Could not create bins for 'actual_rest_len' (min:{min_arl}, max:{max_arl}, bins:{num_bins}): {e_cut}. Trying quantile cut.")
                try:
                    df_bin_analysis['arl_bins'] = pd.qcut(df_bin_analysis['actual_rest_len'], q=min(5, len(df_bin_analysis['actual_rest_len'].unique())-1 ), duplicates='drop')
                    num_bins = df_bin_analysis['arl_bins'].nunique() # Update num_bins
                except Exception as e_qcut:
                    logger_analysis.warning(f"Quantile cut also failed for 'actual_rest_len': {e_qcut}. Skipping this plot.")
                    df_bin_analysis = df_bin_analysis.drop(columns=['arl_bins'], errors='ignore')
            
            if 'arl_bins' in df_bin_analysis.columns:
                error_by_bin_temp_topk = df_bin_analysis.groupby(['arl_bins', 'temp', 'top_k'])['prediction_error'].agg(
                    mean_error='mean'
                ).reset_index()
                error_by_bin_temp_topk['arl_bin_midpoint'] = error_by_bin_temp_topk['arl_bins'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
                error_by_bin_temp_topk.dropna(subset=['arl_bin_midpoint'], inplace=True)
                error_by_bin_temp_topk.sort_values(by=['temp', 'top_k', 'arl_bin_midpoint'], inplace=True)
                
                if not error_by_bin_temp_topk.empty:
                    # Create a FacetGrid for Temp and Top_K
                    unique_temps = sorted(df_bin_analysis['temp'].unique())
                    unique_top_k = sorted(df_bin_analysis['top_k'].unique())
                    
                    if len(unique_temps) > 0 and len(unique_top_k) > 0:
                        g = sns.FacetGrid(error_by_bin_temp_topk, col="top_k", row="temp", 
                                            margin_titles=True, height=4, aspect=1.2, 
                                            col_order=unique_top_k, row_order=unique_temps,
                                            sharex=False, sharey=True) # Share y-axis for easier comparison of error magnitude
                        g.map_dataframe(sns.lineplot, x='arl_bin_midpoint', y='mean_error', marker='o')
                        g.map(plt.axhline, y=0, color='grey', linestyle='--')
                        g.set_axis_labels("Actual Remaining Length (Bin Midpoint)", "Mean Prediction Error")
                        g.set_titles(col_template="TopK = {col_name}", row_template="Temp = {row_name}")
                        g.add_legend()
                        save_plot("Prediction Error vs Actual Rest Len (Faceted by Temp & TopK)", 
                                    output_dir, "error_vs_arl_faceted.png")
                    else:
                        logger_analysis.warning("Not enough unique temp/top_k values for faceting 'Error vs Actual Rest Len'.")
                else:
                    logger_analysis.warning("No data after grouping for 'Error vs Actual Rest Len' faceted plot.")
        else:
            logger_analysis.warning("Insufficient data or range in 'actual_rest_len' for 'Error vs Actual Rest Len' plot.")
    
    # b) Heatmap of MAE by (Temperature, Top_K) - Requires another parameter to be fixed or averaged over
    # For example, fix rep_p and max_new_tok_session, or average MAE over them.
    # This gets complex quickly. Let's do a simpler one: MAE vs Temp, hue by Top_K (already done)
    # Or MAE vs Top_K, hue by Temp:
    if all(c in df.columns for c in ['temp', 'top_k', 'abs_error']):
        mae_vs_topk_temp = df.groupby(['top_k', 'temp'])['abs_error'].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=mae_vs_topk_temp, x='top_k', y='abs_error', hue='temp', marker='o', palette='coolwarm')
        plt.title("MAE vs. Top_K (faceted by Temperature)")
        plt.xlabel("Top_K")
        plt.ylabel("Mean Absolute Error (tokens)")
        plt.legend(title="Temperature")
        plt.grid(True, linestyle='--', alpha=0.7)
        save_plot("MAE vs TopK (by Temperature)", output_dir, "mae_vs_topk_by_temp.png")
    
    # c) Scatter plot of Predicted vs Actual Length (already present, good)
    # Consider adding facets to this as well, e.g., one scatter per (Temp, TopK) group,
    # but this might create too many plots. A sampled overall scatter is a good start.
    # (The scatter plot code from previous version can be kept here)
    if 'predicted_rest_len' in df.columns and 'actual_rest_len' in df.columns:
        sample_df_for_scatter = df.sample(n=min(5000, len(df))) if len(df) > 5000 else df
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=sample_df_for_scatter, x='actual_rest_len', y='predicted_rest_len',
                        hue='prediction_error', size='latency_ms', palette='coolwarm', alpha=0.6,
                        sizes=(20, 200)) # Adjust size range if needed
        max_val_scatter = max(
            sample_df_for_scatter['actual_rest_len'].max(skipna=True), 
            sample_df_for_scatter['predicted_rest_len'].max(skipna=True)
        )
        if pd.notna(max_val_scatter):
            plt.plot([0, max_val_scatter], [0, max_val_scatter], color='red', linestyle='--', label='Perfect Prediction (y=x)')
        plt.title("Predicted vs. Actual Remaining Length (Sampled)")
        plt.xlabel("Actual Remaining Length (tokens)")
        plt.ylabel("Predicted Remaining Length (tokens)")
        # Adjust legend if it gets too crowded
        handles, labels = plt.gca().get_legend_handles_labels()
        # Filter out size legend if too many items, or simplify it
        # For now, keep as is, or plt.legend(title='Prediction Error', handles=handles[:N]) where N is number of hue levels
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        save_plot("Predicted vs Actual Remaining Length (Sampled Scatter)", 
                    output_dir, "predicted_vs_actual_scatter.png")
    logger_analysis.info(f"Analysis complete. Figures saved to {output_dir}")


def plot_length_prediction_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    title: str,
    max_length_for_binning: int = 512, # Consistent with the paper's example range
    num_bins: int = 10
):
    """
    Generates a heatmap comparing binned actual remaining length vs. binned predicted remaining length.
    Cell values represent the count (or log count) of occurrences.
    """
    if df.empty:
        logger_analysis.warning(f"DataFrame is empty, cannot generate length prediction heatmap: {title}.")
        return
    if not all(p in df.columns for p in ['actual_rest_len', 'predicted_rest_len']):
        logger_analysis.warning(
            "Required columns 'actual_rest_len' or 'predicted_rest_len' not in DataFrame for heatmap. Skipping."
        )
        return
    
    df_heatmap = df.copy()
    
    # Ensure data is numeric and handle NaNs
    df_heatmap['actual_rest_len'] = pd.to_numeric(df_heatmap['actual_rest_len'], errors='coerce')
    df_heatmap['predicted_rest_len'] = pd.to_numeric(df_heatmap['predicted_rest_len'], errors='coerce')
    df_heatmap.dropna(subset=['actual_rest_len', 'predicted_rest_len'], inplace=True)
    
    if df_heatmap.empty:
        logger_analysis.warning(f"No valid data after NaN drop for length prediction heatmap: {title}.")
        return
    
    # Clip values to the max_length_for_binning to avoid issues with extreme outliers affecting bins
    # The paper's example bins up to 512.
    df_heatmap['actual_rest_len_clipped'] = np.clip(df_heatmap['actual_rest_len'], 0, max_length_for_binning)
    df_heatmap['predicted_rest_len_clipped'] = np.clip(df_heatmap['predicted_rest_len'], 0, max_length_for_binning)
    
    # Define bin edges (e.g., 10 equal-width bins up to max_length_for_binning)
    # The paper mentions: "The i-th bin bi covers the range [512i/10, 512(i+1)/10)"
    bin_edges = np.linspace(0, max_length_for_binning, num_bins + 1)
    
    # Create labels for the bins (e.g., "b1", "b2", ... or the interval string)
    # For consistency with paper's Y-axis (b10 at top), we might need to reverse bin labels later or adjust plot
    bin_labels = [f'b{i+1}' for i in range(num_bins)] 
    # Or use interval strings:
    # bin_labels_actual = [f"[{bin_edges[i]:.0f}, {bin_edges[i+1]:.0f})" for i in range(num_bins)]
    
    df_heatmap['actual_bin'] = pd.cut(df_heatmap['actual_rest_len_clipped'], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)
    df_heatmap['predicted_bin'] = pd.cut(df_heatmap['predicted_rest_len_clipped'], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)
    
    # Drop rows where binning might have failed (e.g., value exactly on the open upper edge of last bin if right=False not handled perfectly by clip)
    df_heatmap.dropna(subset=['actual_bin', 'predicted_bin'], inplace=True)
    
    if df_heatmap.empty:
        logger_analysis.warning(f"No data after binning for length prediction heatmap: {title}.")
        return
    # Create the contingency table (counts for each pair of bins)
    contingency_table = pd.crosstab(df_heatmap['predicted_bin'], df_heatmap['actual_bin'])
    
    # The paper's Y-axis for predicted length has b10 at the top and b1 at the bottom.
    # pd.crosstab by default sorts index/columns alphabetically (b1, b10, b2...).
    # We need to reorder to match the paper's style or a more natural order.
    # Natural order for bins: b1, b2, ..., b10
    # Paper's Y-axis order: b10, b9, ..., b1 (if we want to match visual exactly)
    
    # Let's use natural bin order for now, b1 at bottom/left.
    # If bin_labels were like "[0-51)", "[51-102)", they'd sort naturally.
    # Since we used "b1", "b2", we need to ensure correct categorical ordering for axes.
    
    # Ensure categorical types for proper ordering in heatmap
    cat_type = pd.CategoricalDtype(categories=bin_labels, ordered=True)
    contingency_table.index = contingency_table.index.astype(cat_type)
    contingency_table.columns = contingency_table.columns.astype(cat_type)
    contingency_table = contingency_table.sort_index(axis=0).sort_index(axis=1)
    
    # Use log scale for counts as in the paper for better visualization of varying frequencies
    # Add 1 before taking log to handle zero counts (log(0) is undefined)
    log_contingency_table = np.log1p(contingency_table)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(log_contingency_table, annot=False, fmt=".1f", cmap="Blues", # Paper uses a blueish map
                linewidths=.5, cbar_kws={'label': 'Log(Count + 1)'}) # annot=False because log values might not be intuitive directly
    
    # To match paper's Y-axis (b10 at top, b1 at bottom), we can reverse the y-axis limits
    # or prepare the contingency table with reversed index.
    # For simplicity, let's keep b1 at bottom for predicted.
    # If you want exact match: contingency_table = contingency_table.iloc[::-1] before heatmap.
    plt.title(title, fontsize=14)
    plt.xlabel("Groundtruth Remaining Length (Binned)", fontsize=12)
    plt.ylabel("Predicted Remaining Length (Binned)", fontsize=12)
    # plt.xticks(rotation=45, ha='right') # May not be needed if bin labels are short
    # plt.yticks(rotation=0)
    
    filename_full = f"{filename}.png"
    save_plot(title, output_dir, filename_full) # Use the helper save_plot


def analyze_and_plot_results_hm(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        logger_analysis.info("DataFrame is empty, no analysis to perform.")
        return
    # ... (Overall Performance, MAE per System Parameter Group, Latency Distribution,
    #      Error vs Actual Rest Length (Faceted), MAE vs TopK, Scatter Plot, etc.
    #      from the previous version of this function can be kept here) ...
    # Ensure 'abs_error' is available if used by other plots
    if 'prediction_error' in df.columns and 'abs_error' not in df.columns:
        df['abs_error'] = df['prediction_error'].abs()
    logger_analysis.info("\n--- Generating Length Prediction Consistency Heatmap (Inspired by TRAIL paper) ---")
    
    # You might want to generate this heatmap for all data, or for specific subsets
    # (e.g., for a particular set of decoding parameters if they significantly affect predictions)
    # For overall consistency:
    plot_length_prediction_heatmap(
        df,
        output_dir=output_dir,
        filename="heatmap_length_pred_consistency_overall",
        title="Overall Consistency: Predicted vs. Ground Truth Remaining Length Bins",
        max_length_for_binning=MAX_MAX_NEW_TOKENS if pd.notna(MAX_MAX_NEW_TOKENS) else 512, # Use your actual max session length for binning
        num_bins=10
    )
    
    # Example: Generate heatmap for a specific temperature (if 'temp' column exists)
    if 'temp' in df.columns:
        common_temp = df['temp'].mode()
        if not common_temp.empty:
            temp_to_filter = common_temp[0]
            df_filtered_by_temp = df[df['temp'] == temp_to_filter]
            if not df_filtered_by_temp.empty:
                plot_length_prediction_heatmap(
                    df_filtered_by_temp,
                    output_dir=output_dir,
                    filename=f"heatmap_length_pred_consistency_temp{temp_to_filter}",
                    title=f"Consistency (Temp={temp_to_filter}): Predicted vs. GT Length Bins",
                    max_length_for_binning=MAX_MAX_NEW_TOKENS if pd.notna(MAX_MAX_NEW_TOKENS) else 512,
                    num_bins=10
                )
            else:
                logger_analysis.warning(f"No data found for temperature {temp_to_filter} to generate specific heatmap.")
    
    logger_analysis.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == '__main__':
    if not RESULTS_JSONL_FILE.exists() or RESULTS_JSONL_FILE.stat().st_size == 0:
        logger_analysis.error(f"Results file for analysis not found or is empty: {RESULTS_JSONL_FILE}")
        logger_analysis.error("Please run the evaluation script first to generate results or update the path.")
    else:
        results_df = load_results_to_dataframe(RESULTS_JSONL_FILE)
        if not results_df.empty:
            # analyze_and_plot_results(results_df, FIGURES_OUTPUT_DIR)
            analyze_and_plot_results_hm(results_df, FIGURES_OUTPUT_DIR)
        else:
            logger_analysis.error("Failed to load data or data is empty after loading. No analysis performed.")