#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --account=liuje-multiai
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=base_v5_seed101
#SBATCH --output=/rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet/outputs/call/base_v5_seed101.out

source ~/.bashrc
conda activate pt311cu130
cd /rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet

python scripts/train.py --config configs/call/base_v5_seed101.yaml --resume-wandb 08mmlr3y --resume-from-checkpoint  outputs/call/base_v5_seed101/checkpoint-94713/ 