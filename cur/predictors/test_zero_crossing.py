#!/usr/bin/env python3
"""
Test script for the new zero-crossing analysis functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from errorRatioEvol import (
    load_and_prepare_data, 
    _log_prompt_analysis_data, 
    _save_parameter_group_log,
    _analyze_error_ratio_zero_crossing
)

def test_zero_crossing_functionality():
    print("=== Testing Zero-Crossing Analysis Functionality ===")
    
    # Load data from one file
    input_file = Path('./pro_length_predictor_eval_results_20250602_214152.jsonl')
    print(f'Loading data from: {input_file}')
    
    if not input_file.exists():
        print(f'Input file not found: {input_file}')
        return
    
    organized_data = load_and_prepare_data(input_file)
    print(f'Loaded {len(organized_data)} parameter groups')
    
    if not organized_data:
        print('No data found')
        return
    
    # Process just the first parameter group with first 10 prompts
    param_key_fset, prompts_data_dict = next(iter(organized_data.items()))
    current_params_dict = dict(param_key_fset)
    print(f'\nProcessing parameter group: {current_params_dict}')
    print(f'Total prompts available: {len(prompts_data_dict)}')
    
    # Create output directory
    test_output_dir = Path('./test_zero_crossing_output')
    test_output_dir.mkdir(exist_ok=True)
    print(f'Output directory: {test_output_dir}')
    
    # Process first 10 prompts only for testing
    param_group_analysis_data = []
    test_prompts_data_dict = {}
    max_prompts = 10
    count = 0
    
    print(f'\nProcessing first {max_prompts} prompts...')
    for prompt_id_str, list_of_steps in prompts_data_dict.items():
        if count >= max_prompts:
            break
        
        print(f'  {count+1}. Processing prompt: {prompt_id_str} ({len(list_of_steps)} steps)')
        
        # Extract analysis data for this prompt
        prompt_analysis = _log_prompt_analysis_data(
            prompt_id=prompt_id_str,
            step_data_list=list_of_steps,
            dec_params=current_params_dict
        )
        param_group_analysis_data.append(prompt_analysis)
        test_prompts_data_dict[prompt_id_str] = list_of_steps
        count += 1
    
    # Test zero-crossing analysis directly
    print(f'\nTesting zero-crossing analysis...')
    zero_crossing_results = _analyze_error_ratio_zero_crossing(
        param_group_analysis_data, 
        test_prompts_data_dict
    )
    
    print(f'Zero-crossing analysis results:')
    print(f'  - Total prompts: {zero_crossing_results["total_prompts"]}')
    print(f'  - Valid fits: {zero_crossing_results["valid_fits_count"]} ({zero_crossing_results["valid_fit_ratio"]:.1%})')
    print(f'  - Accurate zero-crossings: {zero_crossing_results["accurate_zero_crossings_count"]} ({zero_crossing_results["accurate_ratio"]:.1%} of valid fits)')
    
    if zero_crossing_results['accurate_zero_crossings']:
        print(f'\nAccurate zero-crossing examples:')
        for i, item in enumerate(zero_crossing_results['accurate_zero_crossings'][:3], 1):
            print(f'  {i}. {item["prompt_id"]}: slope={item["slope"]:.4f}, intercept={item["intercept"]:.4f}, '
                  f'predicted_step={item["predicted_step"]:.2f}, actual_error={item["actual_error_ratio"]:.3f}')
    
    # Save parameter group log with zero-crossing analysis
    print(f'\nSaving analysis log with zero-crossing analysis...')
    _save_parameter_group_log(
        param_group_data=param_group_analysis_data,
        dec_params=current_params_dict,
        output_dir=test_output_dir,
        step_data_dict=test_prompts_data_dict
    )
    
    # Check if log file was created and show results
    log_file = test_output_dir / "parameter_group_analysis_log.txt"
    if log_file.exists():
        print(f'✓ Log file created successfully: {log_file}')
        print(f'  File size: {log_file.stat().st_size} bytes')
        
        # Show the zero-crossing analysis section
        with open(log_file, 'r') as f:
            content = f.read()
            
        if "ERROR RATIO ZERO-CROSSING ANALYSIS" in content:
            print(f'\n✓ Zero-crossing analysis section found in log file')
            
            # Extract and show the zero-crossing section
            lines = content.split('\n')
            in_zero_crossing_section = False
            zero_crossing_lines = []
            
            for line in lines:
                if "ERROR RATIO ZERO-CROSSING ANALYSIS" in line:
                    in_zero_crossing_section = True
                elif in_zero_crossing_section and line.startswith('='):
                    if zero_crossing_lines:  # End of section
                        break
                
                if in_zero_crossing_section:
                    zero_crossing_lines.append(line)
            
            print(f'\nZero-crossing analysis section:')
            for line in zero_crossing_lines[:20]:  # Show first 20 lines
                print(f'  {line}')
            
            if len(zero_crossing_lines) > 20:
                print(f'  ... (and {len(zero_crossing_lines) - 20} more lines)')
        else:
            print(f'✗ Zero-crossing analysis section not found in log file')
    else:
        print(f'✗ Log file was not created')
    
    print(f'\n=== Test completed successfully! ===')

if __name__ == "__main__":
    test_zero_crossing_functionality()
