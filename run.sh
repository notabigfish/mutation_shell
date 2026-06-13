#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --account=liuje-multiai
#SBATCH --cpus-per-task=36
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a30:1

source ~/.bashrc
conda activate pt311cu130
cd /rds/projects/l/liuje-multiai/shuo/mutation/MuSRNet
python process_data.py --output_dir data/ --pdb_version pdb_260603 --pdb_format mmcif --re_genmatchesm8