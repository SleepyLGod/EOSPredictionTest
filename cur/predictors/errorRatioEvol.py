import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import re
from datetime import datetime


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
ACCURATE_PREDICTION_THRESHOLD = 0.1  # Threshold for considering a prediction "accurate" (|error_ratio| < threshold)
MIN_ACTUAL_REST_LEN_FOR_FITTING = 5  # Minimum actual_rest_len to include in curve fitting


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


def _create_base_plot_with_prediction_and_table(df_plot: pd.DataFrame) -> tuple:
    """Create the base plot with error ratio, prediction length plot, and table below.
    
    Returns:
        tuple: (fig, error_ratio_ax, prediction_ax)
    """
    fig, ax = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1.5, 0.8]})
    error_ratio_ax = ax[0]  # Top subplot for error ratio
    prediction_ax = ax[1]   # Middle subplot for prediction length
    
    if not df_plot.empty:
        sns.lineplot(data=df_plot, x='step_index', y='custom_error_ratio', marker='o', markersize=5, ax=error_ratio_ax, color='darkcyan', label="Custom Error Ratio", zorder=2)
    
    # Add ideal line at y=0 for error ratio
    error_ratio_ax.axhline(0.0, color='red', linestyle='--', linewidth=1.5, label='Ideal Ratio (0.0)', zorder=1)
    
    return fig, error_ratio_ax, prediction_ax


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


def _find_accurate_predictions(df_plot: pd.DataFrame, threshold: float = ACCURATE_PREDICTION_THRESHOLD) -> list:
    """Find the first three decoding steps with accurate predictions (error ratio close to 0)."""
    if df_plot.empty:
        return []
    
    # Find points where |error_ratio| < threshold
    accurate_points = df_plot[df_plot['custom_error_ratio'].abs() < threshold].copy()
    accurate_points = accurate_points.sort_values('step_index')
    
    # Return first three
    return accurate_points.head(3)['step_index'].tolist()


def _add_accurate_prediction_markers(ax, df_plot: pd.DataFrame):
    """Add special markers for the first three accurate predictions."""
    accurate_steps = _find_accurate_predictions(df_plot)
    
    if not accurate_steps:
        return
    
    for i, step_idx in enumerate(accurate_steps):
        step_data = df_plot[df_plot['step_index'] == step_idx]
        if not step_data.empty:
            row = step_data.iloc[0]
            # Add star marker
            ax.scatter(row['step_index'], row['custom_error_ratio'],
                        marker='*', s=200, color='gold', edgecolor='orange',
                        linewidth=2, zorder=5, label=f'Accurate Pred #{i+1}' if i == 0 else "")
            
            # Add text annotation
            ax.text(row['step_index'], row['custom_error_ratio'] + 0.05,
                    f'#{i+1}\nStep {int(step_idx)}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color='darkorange', zorder=6)


def _fit_and_plot_curve(ax, df_plot: pd.DataFrame, dec_params: dict, min_actual_len: int = MIN_ACTUAL_REST_LEN_FOR_FITTING):
    """Fit and plot both linear and quadratic curves for error ratio evolution."""
    if df_plot.empty:
        return
    
    # Get max_new_tokens from decoding parameters
    max_new_tokens = dec_params.get('max_new_tokens', float('inf'))
    
    # Filter data for fitting with stricter criteria
    df_fit = df_plot[
        (df_plot['actual_rest_len'] >= min_actual_len) &  # Exclude small actual_rest_len
        (df_plot['step_index'] + df_plot['predicted_rest_len'] <= max_new_tokens)  # Exclude anomalous predictions
    ].copy()
    
    if len(df_fit) < 3:  # Need at least 3 points for fitting
        logger.info(f"Insufficient data points for fitting after filtering: {len(df_fit)} points")
        return
    
    x_data = df_fit['step_index'].values
    y_data = df_fit['custom_error_ratio'].values
    
    logger.info(f"Fitting curves with {len(df_fit)} points (filtered from {len(df_plot)} total points)")
    
    # Try both linear and quadratic fits
    try:
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        
        # Linear fit
        linear_coeffs = np.polyfit(x_data, y_data, 1)
        linear_poly = np.poly1d(linear_coeffs)
        linear_r2 = np.corrcoef(y_data, linear_poly(x_data))[0, 1] ** 2
        linear_formula = f"y = {linear_coeffs[0]:.4f}x + {linear_coeffs[1]:.4f}"
        
        # Plot linear fit
        y_linear = linear_poly(x_range)
        ax.plot(x_range, y_linear, '--', color='blue', linewidth=2,
                label=f'Linear Fit (R²={linear_r2:.3f})', zorder=3)
        
        # Quadratic fit (if enough points)
        if len(df_fit) >= 5:
            quad_coeffs = np.polyfit(x_data, y_data, 2)
            quad_poly = np.poly1d(quad_coeffs)
            quad_r2 = np.corrcoef(y_data, quad_poly(x_data))[0, 1] ** 2
            quad_formula = f"y = {quad_coeffs[0]:.4f}x² + {quad_coeffs[1]:.4f}x + {quad_coeffs[2]:.4f}"
            
            # Plot quadratic fit
            y_quad = quad_poly(x_range)
            ax.plot(x_range, y_quad, '-.', color='purple', linewidth=2,
                    label=f'Quadratic Fit (R²={quad_r2:.3f})', zorder=3)
            
            # Add formulas annotation
            annotation_text = f"Linear: {linear_formula}\nR² = {linear_r2:.3f}\n\nQuadratic: {quad_formula}\nR² = {quad_r2:.3f}"
        else:
            annotation_text = f"Linear: {linear_formula}\nR² = {linear_r2:.3f}\n\n(Need ≥5 points for quadratic)"
        
        # Add formula annotation
        ax.text(0.02, 0.98, annotation_text,
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='lavender', alpha=0.8))
    
    except (np.linalg.LinAlgError, np.RankWarning) as e:
        logger.warning(f"Curve fitting failed: {e}")
        pass


def _prepare_prediction_length_data(step_data_list: list) -> pd.DataFrame:
    """Prepare data for exact prediction length plot."""
    if not step_data_list:
        return pd.DataFrame()
    
    df_steps = pd.DataFrame(step_data_list)
    
    if df_steps.empty or 'current_full_sequence_len_for_pred' not in df_steps.columns:
        return pd.DataFrame()
    
    # Calculate exact prediction length and ideal length
    df_steps['exact_prediction_length'] = df_steps['current_full_sequence_len_for_pred'] + df_steps['predicted_rest_len']
    df_steps['ideal_length'] = df_steps['current_full_sequence_len_for_pred'] + df_steps['actual_rest_len']
    
    return df_steps


def _plot_prediction_length_evolution(ax, df_steps: pd.DataFrame):
    """Plot exact prediction length evolution with ideal line and fitting curve."""
    if df_steps.empty:
        return
    
    # Plot prediction length evolution
    sns.lineplot(data=df_steps, x='step_index', y='exact_prediction_length',
                marker='o', markersize=4, ax=ax, color='darkgreen',
                label="Exact Prediction Length", zorder=2)
    
    # Calculate and plot ideal length (final actual total length)
    if not df_steps.empty and 'ideal_length' in df_steps.columns:
        # The ideal length should be the final actual total length
        final_actual_length = df_steps['ideal_length'].iloc[-1] if len(df_steps) > 0 else None
        if final_actual_length is not None:
            ax.axhline(final_actual_length, color='red', linestyle='--', linewidth=1.5,
                        label=f'Ideal Length ({final_actual_length:.0f})', zorder=1)
    
    # Fit and plot curve for prediction length
    _fit_prediction_length_curve(ax, df_steps)
    
    # Set labels
    ax.set_xlabel("Decoding Step Index", fontsize=12)
    ax.set_ylabel("Exact Prediction Length", fontsize=12)
    ax.set_title("Exact Prediction Length Evolution", fontsize=14)
    
    # Ensure all points are within axis limits
    if not df_steps.empty:
        y_min = df_steps['exact_prediction_length'].min()
        y_max = df_steps['exact_prediction_length'].max()
        y_range = y_max - y_min
        padding = max(y_range * 0.1, 1)  # At least 1 unit padding
        ax.set_ylim(y_min - padding, y_max + padding)
    
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, linestyle=':', alpha=0.7)


def _fit_prediction_length_curve(ax, df_steps: pd.DataFrame):
    """Fit and plot curve for prediction length evolution."""
    if len(df_steps) < 3:
        return
    
    x_data = df_steps['step_index'].values
    y_data = df_steps['exact_prediction_length'].values
    
    try:
        # Try linear fit first
        linear_coeffs = np.polyfit(x_data, y_data, 1)
        linear_poly = np.poly1d(linear_coeffs)
        linear_r2 = np.corrcoef(y_data, linear_poly(x_data))[0, 1] ** 2
        
        # Try quadratic fit if we have enough points
        if len(df_steps) >= 5:
            quad_coeffs = np.polyfit(x_data, y_data, 2)
            quad_poly = np.poly1d(quad_coeffs)
            quad_r2 = np.corrcoef(y_data, quad_poly(x_data))[0, 1] ** 2
            
            if quad_r2 > linear_r2:
                coeffs = quad_coeffs
                poly = quad_poly
                r2 = quad_r2
                fit_type = "Quadratic"
                formula = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}"
            else:
                coeffs = linear_coeffs
                poly = linear_poly
                r2 = linear_r2
                fit_type = "Linear"
                formula = f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
        else:
            coeffs = linear_coeffs
            poly = linear_poly
            r2 = linear_r2
            fit_type = "Linear"
            formula = f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"
        
        # Plot the fitted curve
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_fitted = poly(x_range)
        
        ax.plot(x_range, y_fitted, '--', color='darkblue', linewidth=2,
                label=f'{fit_type} Fit (R²={r2:.3f})', zorder=3)
        
        # Add formula annotation
        ax.text(0.02, 0.02, f'{fit_type} Fit:\n{formula}\nR² = {r2:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightcyan', alpha=0.8))
    
    except (np.linalg.LinAlgError, np.RankWarning):
        pass


def _extract_error_ratio_curve_formulas(df_plot: pd.DataFrame, dec_params: dict, min_actual_len: int = MIN_ACTUAL_REST_LEN_FOR_FITTING) -> dict:
    """Extract curve formulas for error ratio evolution without plotting.
    Returns:
        dict: Contains linear and quadratic formulas with R² values
    """
    result = {
        'linear_formula': None,
        'linear_r2': None,
        'quadratic_formula': None,
        'quadratic_r2': None,
        'data_points_used': 0
    }
    
    if df_plot.empty:
        return result
    
    # Get max_new_tokens from decoding parameters
    max_new_tokens = dec_params.get('max_new_tokens', float('inf'))
    
    # Filter data for fitting with stricter criteria
    df_fit = df_plot[
        (df_plot['actual_rest_len'] >= min_actual_len) &  # Exclude small actual_rest_len
        (df_plot['step_index'] + df_plot['predicted_rest_len'] <= max_new_tokens)  # Exclude anomalous predictions
    ].copy()
    
    if len(df_fit) < 3:  # Need at least 3 points for fitting
        return result
    
    x_data = df_fit['step_index'].values
    y_data = df_fit['custom_error_ratio'].values
    result['data_points_used'] = len(df_fit)
    
    try:
        # Linear fit
        linear_coeffs = np.polyfit(x_data, y_data, 1)
        linear_poly = np.poly1d(linear_coeffs)
        linear_r2 = np.corrcoef(y_data, linear_poly(x_data))[0, 1] ** 2
        result['linear_formula'] = f"y = {linear_coeffs[0]:.4f}x + {linear_coeffs[1]:.4f}"
        result['linear_r2'] = linear_r2
        
        # Quadratic fit (if enough points)
        if len(df_fit) >= 5:
            quad_coeffs = np.polyfit(x_data, y_data, 2)
            quad_poly = np.poly1d(quad_coeffs)
            quad_r2 = np.corrcoef(y_data, quad_poly(x_data))[0, 1] ** 2
            result['quadratic_formula'] = f"y = {quad_coeffs[0]:.4f}x² + {quad_coeffs[1]:.4f}x + {quad_coeffs[2]:.4f}"
            result['quadratic_r2'] = quad_r2
    
    except (np.linalg.LinAlgError, np.RankWarning) as e:
        logger.warning(f"Error ratio curve fitting failed: {e}")
    
    return result


def _extract_prediction_length_curve_formula(df_steps: pd.DataFrame) -> dict:
    """Extract curve formula for prediction length evolution without plotting.
    Returns:
        dict: Contains best fit formula with R² value
    """
    result = {
        'formula': None,
        'r2': None,
        'fit_type': None,
        'data_points_used': 0
    }
    
    if len(df_steps) < 3:
        return result
    
    x_data = df_steps['step_index'].values
    y_data = df_steps['exact_prediction_length'].values
    result['data_points_used'] = len(df_steps)
    
    try:
        # Try linear fit first
        linear_coeffs = np.polyfit(x_data, y_data, 1)
        linear_poly = np.poly1d(linear_coeffs)
        linear_r2 = np.corrcoef(y_data, linear_poly(x_data))[0, 1] ** 2
        
        # Try quadratic fit if we have enough points
        if len(df_steps) >= 5:
            quad_coeffs = np.polyfit(x_data, y_data, 2)
            quad_poly = np.poly1d(quad_coeffs)
            quad_r2 = np.corrcoef(y_data, quad_poly(x_data))[0, 1] ** 2
            
            if quad_r2 > linear_r2:
                result['formula'] = f"y = {quad_coeffs[0]:.4f}x² + {quad_coeffs[1]:.4f}x + {quad_coeffs[2]:.4f}"
                result['r2'] = quad_r2
                result['fit_type'] = "Quadratic"
            else:
                result['formula'] = f"y = {linear_coeffs[0]:.4f}x + {linear_coeffs[1]:.4f}"
                result['r2'] = linear_r2
                result['fit_type'] = "Linear"
        else:
            result['formula'] = f"y = {linear_coeffs[0]:.4f}x + {linear_coeffs[1]:.4f}"
            result['r2'] = linear_r2
            result['fit_type'] = "Linear"
    
    except (np.linalg.LinAlgError, np.RankWarning) as e:
        logger.warning(f"Prediction length curve fitting failed: {e}")
    
    return result


def _log_prompt_analysis_data(prompt_id: str, step_data_list: list, dec_params: dict) -> dict:
    """Extract and log analysis data for a single prompt.
    Returns:
        dict: Contains all analysis data for the prompt
    """
    if not step_data_list:
        return {
            'prompt_id': prompt_id,
            'total_decoding_steps': 0,
            'error_ratio_curves': {},
            'prediction_length_curve': {},
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Prepare data
    _, df_plot, _ = _prepare_error_ratio_data(step_data_list)
    df_prediction = _prepare_prediction_length_data(step_data_list)
    
    # Extract curve formulas
    error_ratio_curves = _extract_error_ratio_curve_formulas(df_plot, dec_params)
    prediction_length_curve = _extract_prediction_length_curve_formula(df_prediction)
    
    # Calculate total decoding steps
    total_steps = len(step_data_list) if step_data_list else 0
    
    analysis_data = {
        'prompt_id': prompt_id,
        'total_decoding_steps': total_steps,
        'error_ratio_curves': error_ratio_curves,
        'prediction_length_curve': prediction_length_curve,
        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return analysis_data


def _analyze_error_ratio_zero_crossing(param_group_data: list, step_data_dict: dict) -> dict:
    """Analyze error ratio zero-crossing prediction accuracy.
    Args:
        param_group_data: List of analysis data dictionaries for all prompts
        step_data_dict: Dictionary mapping prompt_id to step_data_list for actual data lookup
    Returns:
        dict: Analysis results including statistics and detailed prompt information
    """
    total_prompts = len(param_group_data)
    valid_fits = []
    accurate_zero_crossings = []
    
    for prompt_data in param_group_data:
        prompt_id = prompt_data['prompt_id']
        error_curves = prompt_data['error_ratio_curves']
        
        # Check if linear fit is available and has reasonable quality
        if (error_curves['linear_formula'] and
            error_curves['linear_r2'] is not None and
            error_curves['linear_r2'] >= 0.05 and  # Minimum R² threshold
            error_curves['data_points_used'] >= 5):  # Minimum data points
            
            # Parse linear formula: y = ax + b
            formula = error_curves['linear_formula']
            try:
                # Extract coefficients from formula like "y = -0.0734x + -0.7306"
                parts = formula.replace('y = ', '').replace('x', '').split(' + ')
                if len(parts) == 2:
                    slope = float(parts[0])
                    intercept = float(parts[1])
                elif ' - ' in formula:
                    # Handle negative intercept: "y = -0.0734x + -0.7306" or "y = -0.0734x - 0.7306"
                    formula_clean = formula.replace('y = ', '')
                    if '+ -' in formula_clean:
                        slope_part, intercept_part = formula_clean.split('+ -')
                        slope = float(slope_part.replace('x', '').strip())
                        intercept = -float(intercept_part.strip())
                    elif ' - ' in formula_clean:
                        slope_part, intercept_part = formula_clean.split(' - ')
                        slope = float(slope_part.replace('x', '').strip())
                        intercept = -float(intercept_part.strip())
                    else:
                        continue
                else:
                    continue
                
                # Calculate zero-crossing point: when y = 0, x = -b/a
                if abs(slope) < 1e-6:  # Avoid division by very small numbers
                    continue
                
                predicted_step = -intercept / slope
                
                # Check if predicted step is reasonable (positive and not too large)
                if predicted_step < 0 or predicted_step > 1000:
                    continue
                
                valid_fits.append({
                    'prompt_id': prompt_id,
                    'slope': slope,
                    'intercept': intercept,
                    'r2': error_curves['linear_r2'],
                    'predicted_step': predicted_step,
                    'data_points': error_curves['data_points_used']
                })
                
                # Check actual error ratio at predicted step
                if prompt_id in step_data_dict:
                    step_data_list = step_data_dict[prompt_id]
                    
                    # Find the step closest to predicted_step
                    closest_step_data = None
                    min_distance = float('inf')
                    
                    for step_data in step_data_list:
                        step_index = step_data.get('step_index', 0)
                        distance = abs(step_index - predicted_step)
                        if distance < min_distance:
                            min_distance = distance
                            closest_step_data = step_data
                    
                    if closest_step_data and min_distance <= 2.0:  # Within 2 steps tolerance
                        # Calculate actual error ratio for this step
                        actual_rest_len = closest_step_data.get('actual_rest_len', 0)
                        predicted_rest_len = closest_step_data.get('predicted_rest_len', 0)
                        
                        if actual_rest_len > 0:
                            actual_error_ratio = (predicted_rest_len - actual_rest_len) / actual_rest_len
                        else:
                            actual_error_ratio = float('inf') if predicted_rest_len > 0 else 0.0
                        
                        # Check if actual error ratio is close to 0 (within -0.2 to 0.2)
                        if -0.2 <= actual_error_ratio <= 0.2:
                            accurate_zero_crossings.append({
                                'prompt_id': prompt_id,
                                'slope': slope,
                                'intercept': intercept,
                                'r2': error_curves['linear_r2'],
                                'predicted_step': predicted_step,
                                'actual_step': closest_step_data.get('step_index', 0),
                                'actual_error_ratio': actual_error_ratio,
                                'step_distance': min_distance
                            })
                        
            except (ValueError, IndexError, ZeroDivisionError) as e:
                # Skip prompts with parsing errors
                continue
    
    # Calculate statistics
    valid_fit_ratio = len(valid_fits) / total_prompts if total_prompts > 0 else 0
    accurate_ratio = len(accurate_zero_crossings) / len(valid_fits) if len(valid_fits) > 0 else 0
    
    return {
        'total_prompts': total_prompts,
        'valid_fits_count': len(valid_fits),
        'valid_fit_ratio': valid_fit_ratio,
        'accurate_zero_crossings_count': len(accurate_zero_crossings),
        'accurate_ratio': accurate_ratio,
        'valid_fits': valid_fits,
        'accurate_zero_crossings': accurate_zero_crossings
    }


def _save_parameter_group_log(param_group_data: list, dec_params: dict, output_dir: Path, step_data_dict: dict = None):
    """Save analysis log for a parameter group.
    Args:
        param_group_data: List of analysis data dictionaries for all prompts in this parameter group
        dec_params: Decoding parameters for this group
        output_dir: Output directory for the parameter group
        step_data_dict: Dictionary mapping prompt_id to step_data_list for zero-crossing analysis
    """
    if not param_group_data:
        logger.warning("No data to save for parameter group log")
        return
    
    # Create log filename
    log_filename = "parameter_group_analysis_log.txt"
    log_path = output_dir / log_filename
    
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            # Write header
            param_str = ", ".join([f"{k}={v}" for k, v in sorted(dec_params.items())])
            f.write(f"Parameter Group Analysis Log\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Parameter Group: {param_str}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Prompts: {len(param_group_data)}\n")
            f.write(f"{'=' * 50}\n\n")
            
            # Write data for each prompt
            for i, prompt_data in enumerate(param_group_data, 1):
                f.write(f"Prompt #{i}: {prompt_data['prompt_id']}\n")
                f.write(f"{'-' * 40}\n")
                f.write(f"Total Decoding Steps: {prompt_data['total_decoding_steps']}\n")
                f.write(f"Analysis Timestamp: {prompt_data['analysis_timestamp']}\n\n")
                
                # Error ratio curves
                error_curves = prompt_data['error_ratio_curves']
                f.write("Error Ratio Curves:\n")
                if error_curves['linear_formula']:
                    f.write(f"  - Linear: {error_curves['linear_formula']} (R² = {error_curves['linear_r2']:.3f})\n")
                else:
                    f.write("  - Linear: Not available (insufficient data)\n")
                
                if error_curves['quadratic_formula']:
                    f.write(f"  - Quadratic: {error_curves['quadratic_formula']} (R² = {error_curves['quadratic_r2']:.3f})\n")
                else:
                    f.write("  - Quadratic: Not available (insufficient data)\n")
                
                f.write(f"  - Data points used for fitting: {error_curves['data_points_used']}\n\n")
                
                # Prediction length curve
                pred_curve = prompt_data['prediction_length_curve']
                f.write("Prediction Length Curve:\n")
                if pred_curve['formula']:
                    f.write(f"  - Best Fit ({pred_curve['fit_type']}): {pred_curve['formula']} (R² = {pred_curve['r2']:.3f})\n")
                else:
                    f.write("  - Best Fit: Not available (insufficient data)\n")
                
                f.write(f"  - Data points used for fitting: {pred_curve['data_points_used']}\n")
                f.write(f"\n{'=' * 50}\n\n")
        
        # Perform zero-crossing analysis if step data is provided
        if step_data_dict:
            zero_crossing_analysis = _analyze_error_ratio_zero_crossing(param_group_data, step_data_dict)
            
            # Append zero-crossing analysis to the log file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"ERROR RATIO ZERO-CROSSING ANALYSIS\n")
                f.write(f"{'=' * 60}\n\n")
                
                f.write(f"Total prompts: {zero_crossing_analysis['total_prompts']}\n")
                f.write(f"Prompts with valid linear fits: {zero_crossing_analysis['valid_fits_count']} "
                        f"({zero_crossing_analysis['valid_fit_ratio']:.1%})\n")
                f.write(f"Prompts with accurate zero-crossing prediction: {zero_crossing_analysis['accurate_zero_crossings_count']} "
                        f"({zero_crossing_analysis['accurate_ratio']:.1%} of valid fits)\n\n")
                
                if zero_crossing_analysis['accurate_zero_crossings']:
                    f.write("Accurate Zero-Crossing Prompts:\n")
                    f.write(f"{'Prompt ID':<30} {'Slope':<10} {'Intercept':<10} {'R²':<8} {'Pred.Step':<10} {'Act.Step':<10} {'Act.Error':<10}\n")
                    f.write(f"{'-' * 100}\n")
                    
                    for item in zero_crossing_analysis['accurate_zero_crossings']:
                        f.write(f"{item['prompt_id']:<30} {item['slope']:<10.4f} {item['intercept']:<10.4f} "
                                f"{item['r2']:<8.3f} {item['predicted_step']:<10.2f} {item['actual_step']:<10} "
                                f"{item['actual_error_ratio']:<10.3f}\n")
                
                f.write(f"\n{'=' * 60}\n")
        
        logger.info(f"Parameter group analysis log saved: {log_path}")
    
    except Exception as e:
        logger.error(f"Failed to save parameter group log {log_path}: {e}")


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
    """Add a table below the plots showing the last N steps data."""
    if df_tail.empty:
        return
    
    # Get the third subplot (table area)
    axes = fig.get_axes()
    if len(axes) < 3:
        return
    
    table_ax = axes[2]  # Third subplot for table
    table_ax.axis('off')  # Hide axes
    
    # Prepare table data with short column names
    table_data = []
    headers = ['StepNum', 'ActRLen', 'PredRLen', 'ErrRatio']
    
    for _, row in df_tail.iterrows():
        step_num = int(row['step_index'])
        act_len = int(row['actual_rest_len'])
        pred_len = int(row['predicted_rest_len'])
        err_ratio = row['custom_error_ratio']
        
        # Debug: Log negative predicted lengths
        if pred_len < 0:
            logger.warning(f"Negative predicted_rest_len detected: step={step_num}, pred_len={pred_len}, act_len={act_len}")
        
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
    
    # Prepare prediction length data
    df_prediction = _prepare_prediction_length_data(step_data_list)
    
    # Create base plot with prediction plot and table layout
    fig, error_ratio_ax, prediction_ax = _create_base_plot_with_prediction_and_table(df_plot)
    
    # Add annotations to error ratio plot
    _add_predicted_length_annotations(error_ratio_ax, df_plot)
    _add_special_nan_annotations(error_ratio_ax, df_special_nan_annotations)
    _add_accurate_prediction_markers(error_ratio_ax, df_plot)
    _fit_and_plot_curve(error_ratio_ax, df_plot, dec_params)
    
    # Set labels and title for error ratio plot
    _set_plot_labels_and_title(error_ratio_ax, prompt_id, dec_params)
    
    # Add statistics annotation to error ratio plot
    _add_statistics_annotation(error_ratio_ax, df_plot)
    
    # Finalize error ratio plot
    _finalize_plot(error_ratio_ax, df_plot)
    
    # Plot prediction length evolution
    _plot_prediction_length_evolution(prediction_ax, df_prediction)
    
    # Add tail table below the plots
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
                        k_short = k.replace("temperature", "T").replace("repetition_penalty", "RP").replace("max_new_tokens", "_MNT").replace("top_k", "K")
                        param_group_foldername_parts.append(f"{k_short}{v}")
                    param_group_folder_name = "_".join(param_group_foldername_parts)
                    param_group_output_dir = base_output_dir / sanitize_filename(param_group_folder_name)
                    param_group_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # logger.info(f"Processing parameter group: {current_params_dict} -> saving in {param_group_output_dir}") # Too verbose for many groups
                    
                    # Collect analysis data for this parameter group
                    param_group_analysis_data = []
                    
                    for prompt_id_str, list_of_steps in tqdm(prompts_data_dict.items(), desc=f"Prompts in {param_group_folder_name}", leave=False):
                        # Generate evolution plot with tail table (existing functionality)
                        plot_custom_error_ratio_evolution_for_prompt_annotated(
                            prompt_id=prompt_id_str,
                            step_data_list=list_of_steps,
                            dec_params=current_params_dict,
                            output_dir=param_group_output_dir
                        )
                        
                        # Extract and collect analysis data for logging (new functionality)
                        prompt_analysis = _log_prompt_analysis_data(
                            prompt_id=prompt_id_str,
                            step_data_list=list_of_steps,
                            dec_params=current_params_dict
                        )
                        param_group_analysis_data.append(prompt_analysis)
                    
                    # Save parameter group analysis log with zero-crossing analysis (new functionality)
                    _save_parameter_group_log(
                        param_group_data=param_group_analysis_data,
                        dec_params=current_params_dict,
                        output_dir=param_group_output_dir,
                        step_data_dict=prompts_data_dict
                    )
                
                logger.info("All annotated plotting complete.")
                logger.info(f"Output saved in base directory: {base_output_dir}")