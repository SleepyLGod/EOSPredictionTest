import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import re


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prompt_error_metric_plotter_annotated")

# --- Configuration ---
INPUT_EVAL_JSONL_FILE_LIST = [
    # Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250527_194051.jsonl"), # Databricks
    # Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250526_151530.jsonl"), # Clean
    # Path("./length_predictor_eval_results_Meta_Llama_3_70B_20250529_000246.jsonl"), # Eval
    Path("./pro_length_predictor_eval_results_20250602_214152.jsonl"), # clean pro
    Path("./pro_length_predictor_eval_results_20250603_002030.jsonl"), # databricks pro
    Path("./pro_length_predictor_eval_results_20250603_015311.jsonl"), # eval pro
]
BASE_OUTPUT_DIR_LIST = [
    Path("./eval_output/clean_evol/"),
    Path("./eval_output/databricks_evol/"),
    Path("./eval_output/eval_evol/"),
]

Y_AXIS_LIMITS_CUSTOM_ERROR = (-2.5, 2.5)
ANNOTATE_PREDICTED_LENGTH = True # Control whether to show predicted length annotations
ANNOTATION_FONT_SIZE_PRED_LEN = 6 # Font size for predicted length
ANNOTATION_FONT_SIZE_NAN = 7    # Font size for NaN case predicted length
ANNOTATION_OFFSET_Y = 0.03       # Offset for predicted length text above the point (in data coords)
ERROR_RATIO_ANNOTATION_THRESHOLD = 0.2  # Only annotate points where |error_ratio - 1| < threshold
TAIL_STEPS_COUNT = 10  # Number of final steps to show in table


def sanitize_filename(name: str, max_len: int = 100) -> str:
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'[-\s]+', '-', name)
    return name[:max_len]


def save_plot_to_file(figure, full_path: Path):
    try:
        figure.savefig(full_path, bbox_inches='tight', dpi=150)
        logger.debug(f"Saved plot: {full_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {full_path}: {e}")
    plt.close(figure)


def load_and_prepare_data(jsonl_file_path: Path) -> dict:
    if not jsonl_file_path.exists():
        logger.error(f"Input file not found: {jsonl_file_path}")
        return {}
    data_for_plotting = {}
    sessions_loaded = 0
    sessions_kept = 0
    with open(jsonl_file_path, 'r') as f_in:
        for line_idx, line in enumerate(f_in):
            try:
                session_result = json.loads(line)
                sessions_loaded += 1
                if not session_result.get("eos_encountered_in_session", False):
                    continue
                sessions_kept += 1
                dec_params = session_result.get("decoding_params")
                prompt_id = session_result.get("prompt_id")
                step_predictions = session_result.get("step_predictions")
                if not isinstance(dec_params, dict) or not prompt_id or \
                    not isinstance(step_predictions, list):
                    logger.warning(f"Skipping incomplete session data on line {line_idx + 1}")
                    continue
                processed_steps = []
                for step_data in step_predictions:
                    if isinstance(step_data, dict) and \
                        "step_index" in step_data and \
                        "predicted_rest_len" in step_data and \
                        "actual_rest_len" in step_data:
                        processed_steps.append(step_data.copy())
                    else:
                        logger.debug(f"Skipping malformed step_data in prompt {prompt_id}, line {line_idx+1}")
                if not processed_steps:
                    continue
                param_key = frozenset(sorted(dec_params.items()))
                if param_key not in data_for_plotting:
                    data_for_plotting[param_key] = {}
                data_for_plotting[param_key][prompt_id] = processed_steps
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON on line {line_idx + 1}")
            except Exception as e:
                logger.error(f"Error processing line {line_idx + 1}: {e}")
    logger.info(f"Loaded {sessions_loaded} sessions, kept {sessions_kept} sessions (where EOS was encountered).")
    logger.info(f"Organized data for {len(data_for_plotting)} unique parameter groups.")
    return data_for_plotting


def _prepare_error_ratio_data(step_data_list: list) -> tuple:
    """Prepare and calculate error ratio data from step data list.
    Returns:
        tuple: (df_steps_orig, df_plot, df_special_nan_annotations)
    """
    df_steps_orig = pd.DataFrame(step_data_list) # Keep original for reference
    
    # Calculate the custom error ratio
    df_steps_orig['error_val'] = df_steps_orig['actual_rest_len'] - df_steps_orig['predicted_rest_len']
    
    df_steps_orig['custom_error_ratio'] = np.where(
        np.abs(df_steps_orig['actual_rest_len']) < 1e-9,
        np.where(np.abs(df_steps_orig['predicted_rest_len']) < 0.5, 0.0, np.nan),
        df_steps_orig['error_val'] / df_steps_orig['actual_rest_len']
    )
    
    # Identify special NaN cases (Actual=0, Predicted!=0) BEFORE dropping NaNs for plotting
    df_special_nan_annotations = df_steps_orig[
        (np.abs(df_steps_orig['actual_rest_len']) < 1e-9) &
        (np.abs(df_steps_orig['predicted_rest_len']) >= 0.5) & # Predicted is significantly non-zero
        (df_steps_orig['custom_error_ratio'].isna()) # Ensure it's a NaN that we want to mark
    ].copy() # Use .copy() to avoid SettingWithCopyWarning if we add columns later
    
    # Prepare DataFrame for plotting (remove NaNs for the line plot)
    df_plot = df_steps_orig.dropna(subset=['custom_error_ratio']).copy()
    
    return df_steps_orig, df_plot, df_special_nan_annotations


def _prepare_tail_table_data(step_data_list: list) -> pd.DataFrame:
    """Prepare data for the last N steps table.
    
    Returns:
        DataFrame with columns: step_index, actual_rest_len, predicted_rest_len, custom_error_ratio
    """
    if not step_data_list:
        return pd.DataFrame()
    
    df_steps_orig = pd.DataFrame(step_data_list)
    
    # Get the last N steps
    df_tail = df_steps_orig.tail(TAIL_STEPS_COUNT).copy()
    
    if df_tail.empty:
        return df_tail
    
    # Calculate the custom error ratio
    df_tail['error_val'] = df_tail['actual_rest_len'] - df_tail['predicted_rest_len']
    
    df_tail['custom_error_ratio'] = np.where(
        np.abs(df_tail['actual_rest_len']) < 1e-9,
        np.where(np.abs(df_tail['predicted_rest_len']) < 0.5, 0.0, np.nan),
        df_tail['error_val'] / df_tail['actual_rest_len']
    )
    
    return df_tail


def _create_base_plot_with_table(df_plot: pd.DataFrame) -> tuple:
    """Create the base plot with line and ideal line, with space for table below.
    
    Returns:
        tuple: (fig, ax)
    """
    fig, ax = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    plot_ax = ax[0]  # Top subplot for the plot
    
    if not df_plot.empty:
        sns.lineplot(data=df_plot, x='step_index', y='custom_error_ratio', marker='o', markersize=5, ax=plot_ax, color='darkcyan', label="Custom Error Ratio", zorder=2)
    
    # Add ideal line at y=0
    plot_ax.axhline(0.0, color='red', linestyle='--', linewidth=1.5, label='Ideal Ratio (0.0)', zorder=1)
    
    return fig, plot_ax


def _add_predicted_length_annotations(ax, df_plot: pd.DataFrame):
    """Add predicted length annotations to regular plotted points.
    Only annotate points where error ratio is close to 0 (indicating good predictions).
    """
    if ANNOTATE_PREDICTED_LENGTH and not df_plot.empty:
        for _, row in df_plot.iterrows():
            # Only annotate points where |error_ratio| < threshold (close to ideal prediction)
            if abs(row['custom_error_ratio']) < ERROR_RATIO_ANNOTATION_THRESHOLD:
                ax.text(row['step_index'], row['custom_error_ratio'] + ANNOTATION_OFFSET_Y,
                        f"{row['actual_rest_len']:.0f}", # Show actual remaining length, not predicted
                        color='dimgray', fontsize=ANNOTATION_FONT_SIZE_PRED_LEN,
                        ha='center', va='bottom', zorder=3)


def _add_special_nan_annotations(ax, df_special_nan_annotations: pd.DataFrame):
    """Add annotations for special NaN cases (Actual=0, Predicted!=0)."""
    if not df_special_nan_annotations.empty:
        y_pos_for_nan_annotation = 0.0 # Plot near the ideal line
        if Y_AXIS_LIMITS_CUSTOM_ERROR: # Adjust if y-axis is far from 0
            # Place it slightly inside the bottom limit, or at 0 if 0 is within view
            if 0 < Y_AXIS_LIMITS_CUSTOM_ERROR[0] or 0 > Y_AXIS_LIMITS_CUSTOM_ERROR[1]:
                 y_pos_for_nan_annotation = Y_AXIS_LIMITS_CUSTOM_ERROR[0] + 0.05 * (Y_AXIS_LIMITS_CUSTOM_ERROR[1] - Y_AXIS_LIMITS_CUSTOM_ERROR[0])
        
        for _, row in df_special_nan_annotations.iterrows():
            ax.text(row['step_index'], y_pos_for_nan_annotation,
                    f"{row['predicted_rest_len']:.0f} (*)", # Format as integer, add asterisk
                    color='red', fontsize=ANNOTATION_FONT_SIZE_NAN,
                    ha='center', va='bottom', fontweight='bold', zorder=3,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.2'))


def _set_plot_labels_and_title(ax, prompt_id: str, dec_params: dict):
    """Set plot titles and labels."""
    param_str_title = ", ".join([f"{k.replace('temperature','T').replace('repetition_penalty','RP')}={v}" for k,v in sorted(dec_params.items())])
    ax.set_title(f"Custom Error Ratio Evolution for Prompt: {prompt_id}\nParams: {param_str_title}", fontsize=14)
    ax.set_xlabel("Decoding Step Index", fontsize=12)
    ax.set_ylabel("Error Ratio ((Actual - Pred) / Actual)", fontsize=12)


def _add_statistics_annotation(ax, df_plot: pd.DataFrame):
    """Add statistics annotation box to the plot."""
    if not df_plot.empty:
        mean_ratio = df_plot['custom_error_ratio'].mean()
        median_ratio = df_plot['custom_error_ratio'].median()
        std_ratio = df_plot['custom_error_ratio'].std()
        annotation_text = f"Plotted Points Stats:\nMean Ratio: {mean_ratio:.3f}\nMedian Ratio: {median_ratio:.3f}\nStd Dev: {std_ratio:.3f}\nNum Points: {len(df_plot)}"
        ax.text(0.98, 0.02, annotation_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.5))


def _apply_nonlinear_y_scaling(ax, df_plot: pd.DataFrame):
    """Apply non-linear Y-axis scaling for better visualization of extreme values.
    
    For error ratios >= -2: normal spacing
    For error ratios < -2: compressed spacing
    """
    if df_plot.empty:
        return
    
    min_ratio = df_plot['custom_error_ratio'].min()
    max_ratio = df_plot['custom_error_ratio'].max()
    
    # Only apply non-linear scaling if we have values below -2
    if min_ratio < -2:
        # Create custom y-axis transformation
        def transform_y(y):
            """Transform y values for non-linear scaling"""
            if y >= -2:
                return y  # Normal scaling for y >= -2
            else:
                # Compressed scaling for y < -2: map [-inf, -2] to [-4, -2]
                return -2 - 2 * (1 - np.exp((y + 2) / 5))  # Exponential compression
        
        # Transform the data for plotting
        y_transformed = [transform_y(y) for y in df_plot['custom_error_ratio']]
        
        # Update the plot data
        for line in ax.lines:
            if line.get_label() == "Custom Error Ratio":
                line.set_ydata(y_transformed)
        
        # Set custom y-axis limits and ticks
        y_min_transformed = transform_y(min_ratio)
        y_max_transformed = max(transform_y(max_ratio), 2.5)
        ax.set_ylim(y_min_transformed, y_max_transformed)
        
        # Create custom tick positions and labels
        tick_positions = []
        tick_labels = []
        
        # Normal ticks for y >= -2
        for y in np.arange(-2, y_max_transformed + 0.5, 0.5):
            tick_positions.append(y)
            tick_labels.append(f"{y:.1f}")
        
        # Compressed ticks for y < -2
        extreme_values = [-5, -10, -20, -50]
        for y in extreme_values:
            if y >= min_ratio:
                y_trans = transform_y(y)
                tick_positions.append(y_trans)
                tick_labels.append(f"{y}")
        
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        
        # Add a visual separator at y=-2
        ax.axhline(-2, color='gray', linestyle=':', alpha=0.5, linewidth=1)


def _add_tail_table(fig, df_tail: pd.DataFrame):
    """Add a table below the plot showing the last N steps data."""
    if df_tail.empty:
        return
    
    # Get the second subplot (table area)
    axes = fig.get_axes()
    if len(axes) < 2:
        return
    
    table_ax = axes[1]  # Second subplot for table
    table_ax.axis('off')  # Hide axes
    
    # Prepare table data with short column names
    table_data = []
    headers = ['StepNum', 'ActRLen', 'PredRLen', 'ErrRatio']
    
    for _, row in df_tail.iterrows():
        step_num = int(row['step_index'])
        act_len = int(row['actual_rest_len'])
        pred_len = int(row['predicted_rest_len'])
        err_ratio = row['custom_error_ratio']
        
        # Format error ratio
        if pd.isna(err_ratio):
            err_ratio_str = 'NaN'
        else:
            err_ratio_str = f"{err_ratio:.2f}"
        
        table_data.append([step_num, act_len, pred_len, err_ratio_str])
    
    # Create the table
    table_obj = table_ax.table(cellText=table_data,
                                colLabels=headers,
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 1])
    
    # Style the table
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(9)
    table_obj.scale(1, 1.5)
    
    # Style header row
    for i in range(len(headers)):
        table_obj[(0, i)].set_facecolor('#40466e')
        table_obj[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table_obj[(i, j)].set_facecolor('#f0f0f0')
            else:
                table_obj[(i, j)].set_facecolor('white')
    
    # Add title for the table
    table_ax.set_title(f"Last {TAIL_STEPS_COUNT} Steps Details", fontsize=12, fontweight='bold', pad=10)


def _finalize_plot(ax, df_plot: pd.DataFrame):
    """Apply final plot settings and formatting."""
    # Apply non-linear Y-axis scaling if needed
    _apply_nonlinear_y_scaling(ax, df_plot)
    
    # If no non-linear scaling was applied, use default limits
    if df_plot.empty or df_plot['custom_error_ratio'].min() >= -2:
        if Y_AXIS_LIMITS_CUSTOM_ERROR:
            ax.set_ylim(Y_AXIS_LIMITS_CUSTOM_ERROR)
    
    # Position legend horizontally to avoid vertical stacking
    ax.legend(loc='upper right', ncol=2, columnspacing=1.0)
    ax.grid(True, linestyle=':', alpha=0.7)


def plot_custom_error_ratio_evolution_for_prompt_annotated( # Renamed function
    prompt_id: str,
    step_data_list: list,
    dec_params: dict,
    output_dir: Path
):
    if not step_data_list:
        logger.warning(f"No step data for prompt {prompt_id} with params {dec_params}. Skipping plot.")
        return
    
    # Prepare data
    _, df_plot, df_special_nan_annotations = _prepare_error_ratio_data(step_data_list)
    
    if df_plot.empty and df_special_nan_annotations.empty: # No data to plot at all
        logger.info(f"No valid custom error ratios or special NaN cases to plot for prompt {prompt_id}, params {dec_params}")
        return
    
    # Prepare tail table data
    df_tail = _prepare_tail_table_data(step_data_list)
    
    # Create base plot with table layout
    fig, ax = _create_base_plot_with_table(df_plot)
    
    # Add annotations
    _add_predicted_length_annotations(ax, df_plot)
    _add_special_nan_annotations(ax, df_special_nan_annotations)
    
    # Set labels and title
    _set_plot_labels_and_title(ax, prompt_id, dec_params)
    
    # Add statistics annotation
    _add_statistics_annotation(ax, df_plot)
    
    # Finalize plot
    _finalize_plot(ax, df_plot)
    
    # Add tail table below the plot
    _add_tail_table(fig, df_tail)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    prompt_filename_safe = sanitize_filename(prompt_id)
    plot_filename = f"prompt_{prompt_filename_safe}_custom_err_ratio_evol_annot.png"
    full_plot_path = output_dir / plot_filename
    save_plot_to_file(fig, full_plot_path)


# --- Main Execution ---
if __name__ == "__main__":
    for input_file, base_output_dir in zip(INPUT_EVAL_JSONL_FILE_LIST, BASE_OUTPUT_DIR_LIST): 
        logger.info(f"Starting script to plot annotated custom error ratio evolution from: {input_file}")
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        if str(input_file).startswith("Path/To/Your"):
            logger.critical("CRITICAL: Please update 'INPUT_EVAL_JSONL_FILE' with the correct path to your log file.")
        else:
            organized_data = load_and_prepare_data(input_file)
            
            if not organized_data:
                logger.warning("No data loaded or prepared. Exiting.")
            else:
                logger.info("Starting annotated plot generation...")
                for param_key_fset, prompts_data_dict in tqdm(organized_data.items(), desc="Parameter Groups"):
                    current_params_dict = dict(param_key_fset)
                    
                    param_group_foldername_parts = []
                    for k, v in sorted(current_params_dict.items()):
                        k_short = k.replace("temperature", "T").replace("repetition_penalty", "RP").replace("max_new_tokens", "MNT").replace("top_k", "K")
                        param_group_foldername_parts.append(f"{k_short}{v}")
                    param_group_folder_name = "_".join(param_group_foldername_parts)
                    param_group_output_dir = base_output_dir / sanitize_filename(param_group_folder_name)
                    param_group_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # logger.info(f"Processing parameter group: {current_params_dict} -> saving in {param_group_output_dir}") # Too verbose for many groups
                    
                    for prompt_id_str, list_of_steps in tqdm(prompts_data_dict.items(), desc=f"Prompts in {param_group_folder_name}", leave=False):
                        # Generate evolution plot with tail table
                        plot_custom_error_ratio_evolution_for_prompt_annotated(
                            prompt_id=prompt_id_str,
                            step_data_list=list_of_steps,
                            dec_params=current_params_dict,
                            output_dir=param_group_output_dir
                        )
                
                logger.info("All annotated plotting complete.")
                logger.info(f"Output saved in base directory: {base_output_dir}")