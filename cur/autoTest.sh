#!/bin/bash

# Define parameter values (a: 2,4,5,6,7; b: fixed at 2; c: 0-7)
a_values=(2 4 5 6 7)
b=2
c_values=$(seq 0 7)

# Default model selection (1-7 based on your interface)
model_choice=1

# Main test loop
for a in "${a_values[@]}"; do
  for c in $c_values; do
    echo "[INFO] Testing combination: a=$a, b=$b, c=$c, model=$model_choice"
    
    # Generate multi-line input with explicit newlines
    # Input order must match Python script's input() sequence:
    # 1. a  2. b  3. c  4. model_choice
    input_data=$(printf "%d\n%d\n%d\n%d\n" "$a" "$b" "$c" "$model_choice")
    
    # Execute with timing and input redirection
    time python mapTest.py <<< "$input_data"
    
    # Error handling and status reporting
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] Failed at a=$a, c=$c (exit code: $exit_code)"
        # exit 1  # Uncomment to abort on first error
    fi
    
    echo "---------------------------------"
  done
done
