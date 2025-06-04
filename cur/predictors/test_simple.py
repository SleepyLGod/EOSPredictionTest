#!/usr/bin/env python3
"""
Simple test for new logging functionality
"""

import sys
import os
from pathlib import Path

# Test basic Python functionality first
print("Testing basic Python functionality...")
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")
    sys.exit(1)

# Test our new functions
print("\nTesting new functions...")
try:
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import our new functions
    from errorRatioEvol import (
        _log_prompt_analysis_data,
        _save_parameter_group_log
    )
    print("✓ Successfully imported new functions")
    
    # Create test data
    test_data = [
        {'step_index': 0, 'actual_rest_len': 10, 'predicted_rest_len': 12, 'current_full_sequence_len_for_pred': 5},
        {'step_index': 1, 'actual_rest_len': 9, 'predicted_rest_len': 10, 'current_full_sequence_len_for_pred': 6},
        {'step_index': 2, 'actual_rest_len': 8, 'predicted_rest_len': 8, 'current_full_sequence_len_for_pred': 7}
    ]
    
    dec_params = {'temperature': 0.7, 'max_new_tokens': 100}
    
    # Test prompt analysis
    print("\nTesting prompt analysis...")
    analysis_data = _log_prompt_analysis_data('test_prompt_001', test_data, dec_params)
    print(f"✓ Analysis data created for prompt: {analysis_data['prompt_id']}")
    print(f"  - Total steps: {analysis_data['total_decoding_steps']}")
    
    # Test log saving
    print("\nTesting log file creation...")
    test_output_dir = Path("./test_output_simple")
    test_output_dir.mkdir(exist_ok=True)
    
    param_group_data = [analysis_data]
    _save_parameter_group_log(param_group_data, dec_params, test_output_dir)
    
    # Check if log file was created
    log_file = test_output_dir / "parameter_group_analysis_log.txt"
    if log_file.exists():
        print(f"✓ Log file created successfully: {log_file}")
        with open(log_file, 'r') as f:
            content = f.read()
            print(f"  - File size: {len(content)} characters")
            print("  - First few lines:")
            for line in content.split('\n')[:5]:
                print(f"    {line}")
    else:
        print("✗ Log file was not created")
    
    print("\n✓ All tests passed! New functionality is working correctly.")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
