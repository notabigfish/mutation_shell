#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --account=liuje-multiai
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=smoke_base_v5_v2
#SBATCH --output=/rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet/outputs/call/smoke_base_v5_v2.out

source ~/.bashrc
conda activate pt311cu130
cd /rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet

python scripts/train.py --config configs/call/smoke_base_v5_v2.yaml