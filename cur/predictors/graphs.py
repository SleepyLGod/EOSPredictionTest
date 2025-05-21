import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger_analysis = logging.getLogger("analysis") # Separate logger for analysis script

RESULTS_JSONL_FILE = Path("./eval_output/length_predictor_eval_results_Meta_Llama_3_70B_20250521_010529.jsonl") 
# "length_predictor_eval_results_Meta_Llama_3_70B_20250521_140235.jsonl"
FIGURES_OUTPUT_DIR = Path("./results/")
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_results_to_dataframe(jsonl_file_path: Path) -> pd.DataFrame:
    """Loads the JSONL results file into a pandas DataFrame, expanding step_predictions."""
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
                # Convert dict to a hashable type (tuple of items) for groupby if needed later for params
                # dec_params_tuple = tuple(sorted(dec_params.items())) 

                for step_detail in session_result.get("step_predictions", []):
                    flat_step = {
                        "prompt_id": prompt_id,
                        # Flatten decoding_params into individual columns for easier filtering/grouping
                        "temp": dec_params.get("temperature"),
                        "top_k": dec_params.get("top_k"),
                        "rep_p": dec_params.get("repetition_penalty"),
                        "max_new_tok_session": dec_params.get("max_new_tokens"),
                        "actual_generated_steps_session": session_result.get("actual_generated_steps"),
                        "eos_encountered_session": session_result.get("eos_encountered_in_session"),
                        **step_detail # Add all keys from step_detail
                    }
                    all_steps_data.append(flat_step)
            except json.JSONDecodeError:
                logger_analysis.warning(f"Skipping invalid JSON line {line_idx+1} in {jsonl_file_path}")
                continue
            except Exception as e:
                logger_analysis.error(f"Error processing line {line_idx+1}: {e}")
                continue
    
    if not all_steps_data:
        logger_analysis.warning("No data loaded from results file.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_steps_data)
    logger_analysis.info(f"Loaded {len(df)} step prediction records into DataFrame.")
    # Optional: Convert relevant columns to numeric if they aren't already
    numeric_cols = ['predicted_rest_len', 'actual_rest_len', 'latency_ms', 'prediction_error', 
                    'current_full_sequence_len_for_pred', 'step_index', 
                    'temp', 'top_k', 'rep_p', 'max_new_tok_session', 'actual_generated_steps_session']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN
    
    return df

def analyze_and_plot_results(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        logger_analysis.info("DataFrame is empty, no analysis to perform.")
        return

    # --- 1. 每个prompt在每个decoding step的预测偏差等数据 ---
    # 这个数据已经在 DataFrame 'df' 中了。我们可以选择性地打印或保存。
    # 例如，打印前几个 prompt 的前几个 step 的详细信息
    logger_analysis.info("\n--- Sample of Per-Step Prediction Data (First few prompts/steps) ---")
    sample_data_to_show = df.groupby("prompt_id").head(3).head(15) # Show first 3 steps of first 5 (3*5=15) unique prompt_id groups
    logger_analysis.info(sample_data_to_show[[
        "prompt_id", "step_index", "temp", "top_k", "predicted_rest_len", 
        "actual_rest_len", "prediction_error", "latency_ms"
    ]].to_string())

    # --- 2. 每个system parameter group下的图表分析 ---
    # 我们需要定义什么是 "system parameter group"。通常是 (temp, top_k, rep_p, max_new_tok_session) 的组合。
    
    # Create a combined parameter group string for easier grouping if needed
    # but individual parameter columns are often more useful for plotting.
    # df['param_group_str'] = df.apply(
    #     lambda row: f"T{row['temp']}_K{row['top_k']}_RP{row['rep_p']}_MNT{row['max_new_tok_session']}", 
    #     axis=1
    # )

    # a) MAE per parameter group
    logger_analysis.info("\n--- MAE per System Parameter Group ---")
    param_cols = ['temp', 'top_k', 'rep_p', 'max_new_tok_session']
    # Ensure no NaN in prediction_error before abs()
    df_cleaned_error = df.dropna(subset=['prediction_error'])
    mae_per_group = df_cleaned_error.groupby(param_cols).apply(
        lambda x: mean_absolute_error(x['actual_rest_len'], x['predicted_rest_len'])
    ).sort_values()
    logger_analysis.info(mae_per_group.to_string())
    
    # Plot MAE for top N worst/best groups (example)
    if not mae_per_group.empty:
        plt.figure(figsize=(12, 7))
        # Convert multi-index to string for plotting
        mae_per_group.head(10).plot(kind='barh', title='Top 10 Best MAE by Parameter Group')
        plt.xlabel("Mean Absolute Error (tokens)")
        plt.ylabel("Parameter Group (T, K, RP, MNT)")
        plt.tight_layout()
        plt.savefig(output_dir / "mae_best_param_groups.png")
        plt.close()

        plt.figure(figsize=(12, 7))
        mae_per_group.tail(10).plot(kind='barh', title='Top 10 Worst MAE by Parameter Group')
        plt.xlabel("Mean Absolute Error (tokens)")
        plt.ylabel("Parameter Group (T, K, RP, MNT)")
        plt.tight_layout()
        plt.savefig(output_dir / "mae_worst_param_groups.png")
        plt.close()

    # b) Latency distribution per parameter group (e.g., by temperature)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='temp', y='latency_ms', hue='top_k') # Example: latency vs temp, colored by top_k
    plt.title("Length Predictor Latency Distribution by Temperature and Top_K")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Temperature")
    plt.yscale('log') # Latency can have wide range
    plt.tight_layout()
    plt.savefig(output_dir / "latency_by_temp_topk.png")
    plt.close()
    logger_analysis.info("Generated latency_by_temp_topk.png")

    # --- 3. 不同system parameter情况下的对比分析 ---

    # a) Prediction Error vs. Actual Rest Length (faceted by a parameter, e.g., temperature)
    if 'actual_rest_len' in df.columns and 'prediction_error' in df.columns and 'temp' in df.columns:
        # Create bins for actual_rest_len for better visualization
        max_arl = df['actual_rest_len'].max()
        if pd.notna(max_arl) and max_arl > 0 :
            bins = pd.cut(df['actual_rest_len'], bins=np.arange(0, max_arl + 50, 50), right=False) # Bins of 50
            
            plt.figure(figsize=(14, 8))
            sns.lineplot(data=df, x=bins, y='prediction_error', hue='temp', errorbar=('ci', 95), estimator=np.mean)
            plt.title("Mean Prediction Error vs. Actual Remaining Length (by Temperature)")
            plt.xlabel("Actual Remaining Length (Binned)")
            plt.ylabel("Mean Prediction Error (Pred - Actual)")
            plt.xticks(rotation=45, ha='right')
            plt.axhline(0, color='grey', linestyle='--') # Line for zero error
            plt.tight_layout()
            plt.savefig(output_dir / "error_vs_actual_rest_len_by_temp.png")
            plt.close()
            logger_analysis.info("Generated error_vs_actual_rest_len_by_temp.png")
        else:
            logger_analysis.warning("Could not generate 'Error vs. Actual Rest Length' plot due to missing or invalid 'actual_rest_len' data.")


    # b) MAE vs. Temperature (with top_k as hue)
    if all(c in df_cleaned_error.columns for c in ['temp', 'top_k', 'actual_rest_len', 'predicted_rest_len']):
        # Create a temporary column for absolute error for MAE calculation
        df_cleaned_error['abs_error'] = np.abs(df_cleaned_error['prediction_error'])
        mae_vs_temp_topk = df_cleaned_error.groupby(['temp', 'top_k'])['abs_error'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=mae_vs_temp_topk, x='temp', y='abs_error', hue='top_k', marker='o', palette='viridis')
        plt.title("MAE vs. Temperature (faceted by Top_K)")
        plt.xlabel("Temperature")
        plt.ylabel("Mean Absolute Error (tokens)")
        plt.legend(title="Top_K")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "mae_vs_temp_by_topk.png")
        plt.close()
        logger_analysis.info("Generated mae_vs_temp_by_topk.png")
    else:
        logger_analysis.warning("Could not generate 'MAE vs. Temperature' plot due to missing columns.")


    # c) Scatter plot of Predicted vs Actual Length, colored by error magnitude
    if 'predicted_rest_len' in df.columns and 'actual_rest_len' in df.columns and 'prediction_error' in df.columns:
        # Sample a subset if dataframe is too large for a clear scatter plot
        sample_df_for_scatter = df.sample(n=min(5000, len(df))) if len(df) > 5000 else df
        
        plt.figure(figsize=(10, 8))
        scatter_plot = sns.scatterplot(
            data=sample_df_for_scatter, 
            x='actual_rest_len', 
            y='predicted_rest_len',
            hue='prediction_error', # Color by error
            size='latency_ms', # Optional: size by latency
            palette='coolwarm', # Diverging palette for error
            alpha=0.6
        )
        # Add y=x line for reference
        max_val = max(sample_df_for_scatter['actual_rest_len'].max(), sample_df_for_scatter['predicted_rest_len'].max())
        if pd.notna(max_val):
            plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction (y=x)')
        plt.title("Predicted vs. Actual Remaining Length (Sampled)")
        plt.xlabel("Actual Remaining Length (tokens)")
        plt.ylabel("Predicted Remaining Length (tokens)")
        plt.legend(title='Prediction Error')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "predicted_vs_actual_scatter.png")
        plt.close()
        logger_analysis.info("Generated predicted_vs_actual_scatter.png")
    else:
        logger_analysis.warning("Could not generate 'Predicted vs. Actual Scatter' plot due to missing columns.")

    logger_analysis.info(f"Analysis complete. Figures saved to {output_dir}")


if __name__ == '__main__':
    # This is an example of how you would run the analysis.
    # Ensure RESULTS_JSONL_FILE points to your actual results file.
    if not RESULTS_JSONL_FILE.exists():
        logger_analysis.error(f"Results file for analysis not found: {RESULTS_JSONL_FILE}")
        logger_analysis.error("Please run the evaluation script first or update the path.")
    else:
        results_df = load_results_to_dataframe(RESULTS_JSONL_FILE)
        if not results_df.empty:
            analyze_and_plot_results(results_df, FIGURES_OUTPUT_DIR)
        else:
            logger_analysis.error("Failed to load data or data is empty. No analysis performed.")