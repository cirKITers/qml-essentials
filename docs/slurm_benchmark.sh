#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=qml-essentials
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=10
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=06:00:00
# 
# partition the job will run on
#SBATCH --partition cpu
# 
# expected memory requirements
#SBATCH --mem=200GB
#
# infos
#
# output path
#SBATCH --output="logs/slurm/slurm-%j-%x.out"

module load compiler/llvm
module load devel/python/3.12.3

cd ~/qml-essentials
uv run python docs/benchmarks.py
