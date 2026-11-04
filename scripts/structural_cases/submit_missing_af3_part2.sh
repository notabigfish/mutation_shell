#!/bin/bash
set -euo pipefail

jobs_file="results/structural_cases/af3_jobs.txt"
sbatch_script="scripts/structural_cases/run_af3-part2.sbatch"

missing=(
    pair0004_mut pair0004_wt
)

declare -A wanted
for name in "${missing[@]}"; do
    wanted["$name"]=1
done

indices=()
idx=0

while IFS= read -r json || [[ -n "$json" ]]; do
    name=$(basename "$json" .json)

    if [[ -n "${wanted[$name]+x}" ]]; then
        indices+=("$idx")
        echo "Task $idx -> $name"
        unset "wanted[$name]"
    fi

    ((idx += 1))
done < "$jobs_file"

# Ensure all requested predictions were found
if (( ${#wanted[@]} > 0 )); then
    echo "ERROR: Missing entries in $jobs_file:"
    printf '%s\n' "${!wanted[@]}"
    exit 1
fi

# Submit only the selected task IDs
array_ids=$(IFS=,; echo "${indices[*]}")

echo "Submitting ${#indices[@]} tasks..."
sbatch --array="${array_ids}%4" "$sbatch_script"