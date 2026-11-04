#!/bin/bash
export SLURM_SUBMIT_DIR="$PWD"
export SLURM_ARRAY_JOB_ID="manual_$(date +%Y%m%d_%H%M%S)"

jobs_file="results/structural_cases/af3_jobs.txt"
script="scripts/structural_cases/run_af3.sbatch"

missing=(
    pair0094_mut pair0094_wt
    pair0170_mut pair0170_wt
)

# Map names to 0-based SLURM task IDs
declare -A task_ids
idx=0

while IFS= read -r json || [[ -n "$json" ]]; do
    name=$(basename "$json" .json)
    task_ids["$name"]=$idx
    ((idx += 1))
done < "$jobs_file"

# Validate before running
for name in "${missing[@]}"; do
    if [[ ! -v task_ids[$name] ]]; then
        echo "ERROR: $name not found in $jobs_file"
        exit 1
    fi
done

# Run sequentially on the current node
for name in "${missing[@]}"; do
    task_id=${task_ids[$name]}

    echo "Running task $task_id: $name"

    SLURM_ARRAY_TASK_ID="$task_id" bash "$script"

    echo "Finished: $name"
done

echo "All missing tasks completed."