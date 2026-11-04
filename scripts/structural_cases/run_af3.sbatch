#!/bin/bash
#SBATCH --output=results/structural_cases/af3_%A_%a.out
#SBATCH --qos=bbgpu
#SBATCH --account=liuje-multiai
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=12:0:0
#SBATCH --mem=128G
#SBATCH --job-name=MuSR-AF3
#SBATCH --array=0-300%4
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
job_json=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/structural_cases/af3_jobs.txt)
name=$(basename "$job_json" .json)
output=results/structural_cases/af3_predictions
if [ -s "$output/$name/${name}_model.cif" ]; then
    exit 0
fi

HOST_TMP=/rds/projects/l/liuje-multiai/shuo/tmp
singularity exec --nv \
    --bind "$HOST_TMP:/af3_tmp" \
    --env TMPDIR=/af3_tmp \
    /rds/bear-apps/container-images/singularity/x86_64/a/AlphaFold3/alphafold3-3.0.0.sif \
    python /app/alphafold/run_alphafold.py \
        --json_path="$(realpath "$job_json")" \
        --model_dir=/rds/projects/l/liuje-multiai/shuo/dwnl_ckpts \
        --output_dir="$(realpath "$output")" \
        --jackhmmer_n_cpu=4 2>&1 | tee "results/structural_cases/af3_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
