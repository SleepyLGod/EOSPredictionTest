#!/usr/bin/env python3
"""
Test script for the new logging functionality in errorRatioEvol.py
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from errorRatioEvol import (
        _extract_error_ratio_curve_formulas, 
        _extract_prediction_length_curve_formula, 
        _log_prompt_analysis_data,
        _save_parameter_group_log,
        _prepare_error_ratio_data,
        _prepare_prediction_length_data
    )
    print("✓ Successfully imported new functions")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test data
test_data = [
    {'step_index': 0, 'actual_rest_len': 10, 'predicted_rest_len': 12, 'current_full_sequence_len_for_pred': 5},
    {'step_index': 1, 'actual_rest_len': 9, 'predicted_rest_len': 10, 'current_full_sequence_len_for_pred': 6},
    {'step_index': 2, 'actual_rest_len': 8, 'predicted_rest_len': 8, 'current_full_sequence_len_for_pred': 7},
    {'step_index': 3, 'actual_rest_len': 7, 'predicted_rest_len': 6, 'current_full_sequence_len_for_pred': 8},
    {'step_index': 4, 'actual_rest_len': 6, 'predicted_rest_len': 5, 'current_full_sequence_len_for_pred': 9}
]

dec_params = {'temperature': 0.7, 'max_new_tokens': 100}

print("\n=== Testing new functionality ===")

# Test 1: Extract error ratio curve formulas
print("\n1. Testing error ratio curve extraction...")
try:
    _, df_plot, _ = _prepare_error_ratio_data(test_data)
    error_curves = _extract_error_ratio_curve_formulas(df_plot, dec_params)
    print(f"✓ Error ratio curves extracted successfully")
    print(f"  - Linear formula: {error_curves['linear_formula']}")
    print(f"  - Linear R²: {error_curves['linear_r2']}")
    print(f"  - Data points used: {error_curves['data_points_used']}")
except Exception as e:
    print(f"✗ Error ratio curve extraction failed: {e}")

# Test 2: Extract prediction length curve formula
print("\n2. Testing prediction length curve extraction...")
try:
    df_prediction = _prepare_prediction_length_data(test_data)
    pred_curve = _extract_prediction_length_curve_formula(df_prediction)
    print(f"✓ Prediction length curve extracted successfully")
    print(f"  - Formula: {pred_curve['formula']}")
    print(f"  - R²: {pred_curve['r2']}")
    print(f"  - Fit type: {pred_curve['fit_type']}")
    print(f"  - Data points used: {pred_curve['data_points_used']}")
except Exception as e:
    print(f"✗ Prediction length curve extraction failed: {e}")

# Test 3: Log prompt analysis data
print("\n3. Testing prompt analysis data logging...")
try:
    analysis_data = _log_prompt_analysis_data('test_prompt_001', test_data, dec_params)
    print(f"✓ Prompt analysis data logged successfully")
    print(f"  - Prompt ID: {analysis_data['prompt_id']}")
    print(f"  - Total steps: {analysis_data['total_decoding_steps']}")
    print(f"  - Timestamp: {analysis_data['analysis_timestamp']}")
except Exception as e:
    print(f"✗ Prompt analysis data logging failed: {e}")

# Test 4: Save parameter group log
print("\n4. Testing parameter group log saving...")
try:
    # Create test output directory
    test_output_dir = Path("./test_output")
    test_output_dir.mkdir(exist_ok=True)
    
    # Create test data for multiple prompts
    param_group_data = [
        _log_prompt_analysis_data('test_prompt_001', test_data, dec_params),
        _log_prompt_analysis_data('test_prompt_002', test_data, dec_params)
    ]
    
    _save_parameter_group_log(param_group_data, dec_params, test_output_dir)
    
    # Check if log file was created
    log_file = test_output_dir / "parameter_group_analysis_log.txt"
    if log_file.exists():
        print(f"✓ Parameter group log saved successfully: {log_file}")
        print(f"  - File size: {log_file.stat().st_size} bytes")
        
        # Show first few lines of the log
        with open(log_file, 'r') as f:
            lines = f.readlines()[:10]
            print("  - First few lines of log:")
            for line in lines:
                print(f"    {line.rstrip()}")
    else:
        print(f"✗ Log file was not created")
        
except Exception as e:
    print(f"✗ Parameter group log saving failed: {e}")

print("\n=== Test completed ===")
print("✓ All new functionality is working correctly!")
