#!/bin/bash

# Define parameter combinations
a_values=(2 4 5 6 7)       # Allowed values for parameter 'a'
b=2                        # Fixed value for parameter 'b'
c_values=$(seq 0 7)        # Sequence for parameter 'c' (0-7 inclusive)
model_choices=(1)          # Default model selection (expand to (1 2 3 4 5 6 7) for multiple models)

# Create log directory with timestamp
log_dir="test_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# Generate task list file with all combinations
task_list="task_list.txt"
true > "$task_list"  # Clear previous task list

# Generate all parameter combinations (a, b, c, model)
for model in "${model_choices[@]}"; do
  for a in "${a_values[@]}"; do
    for c in $c_values; do
      echo "$a $b $c $model" >> "$task_list"
    done
  done
done

# Parallel execution parameters
max_jobs=4                  # Concurrent processes (adjust based on resources)
job_timeout=$((10*3600))    # Per-task timeout in seconds (10 hours)

# Execute using GNU Parallel with proper input handling
parallel --jobs $max_jobs \
         --timeout $job_timeout \
         --joblog "${log_dir}/joblog.csv" \
         --progress \
         --resume-failed \
         --eta \
         --colsep ' ' \
         "printf '%d\\n%d\\n%d\\n%d\\n' {1} {2} {3} {4} | python3 mapTest.py > ${log_dir}/a_{1}_b_{2}_c_{3}_model_{4}.log 2>&1" \
         :::: "$task_list"

# Cleanup and summary
rm -f "$task_list"
echo "[STATUS] All jobs completed. Logs stored in: $log_dir"
