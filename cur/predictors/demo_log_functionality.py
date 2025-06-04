#!/usr/bin/env python3
"""
Demo script to show the new logging functionality
This script processes a small subset of data to generate log files
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
    sanitize_filename
)

def main():
    print("=== Demo: New Logging Functionality ===")
    
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
    
    # Process just the first parameter group with first 5 prompts
    param_key_fset, prompts_data_dict = next(iter(organized_data.items()))
    current_params_dict = dict(param_key_fset)
    print(f'\nProcessing parameter group: {current_params_dict}')
    print(f'Total prompts available: {len(prompts_data_dict)}')
    
    # Create output directory
    param_group_foldername_parts = []
    for k, v in sorted(current_params_dict.items()):
        k_short = k.replace("temperature", "T").replace("repetition_penalty", "RP").replace("max_new_tokens", "_MNT").replace("top_k", "K")
        param_group_foldername_parts.append(f"{k_short}{v}")
    param_group_folder_name = "_".join(param_group_foldername_parts)
    
    demo_output_dir = Path('./demo_log_output') / sanitize_filename(param_group_folder_name)
    demo_output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {demo_output_dir}')
    
    # Process first 5 prompts only for demo
    param_group_analysis_data = []
    max_prompts = 5
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
        count += 1
    
    # Save parameter group log
    print(f'\nSaving analysis log...')
    _save_parameter_group_log(
        param_group_data=param_group_analysis_data,
        dec_params=current_params_dict,
        output_dir=demo_output_dir
    )
    
    # Show results
    log_file = demo_output_dir / "parameter_group_analysis_log.txt"
    if log_file.exists():
        print(f'✓ Log file created successfully: {log_file}')
        print(f'  File size: {log_file.stat().st_size} bytes')
        
        # Show first few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f'\nFirst 15 lines of the log file:')
            for i, line in enumerate(lines[:15], 1):
                print(f'{i:2d}: {line.rstrip()}')
            
            if len(lines) > 15:
                print(f'... (and {len(lines) - 15} more lines)')
    else:
        print('✗ Log file was not created')
    
    print(f'\n=== Demo completed successfully! ===')
    print(f'The new functionality adds analysis logs to each parameter group folder.')
    print(f'Each log contains:')
    print(f'  - Parameter group information')
    print(f'  - For each prompt: ID, total decoding steps, curve formulas, R² values')
    print(f'  - Timestamps for analysis')

if __name__ == "__main__":
    main()
