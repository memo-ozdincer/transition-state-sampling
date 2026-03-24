#!/bin/bash
#SBATCH --job-name=phase3_300s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/phase3_300s_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/phase3_300s_%j.err
#SBATCH --account=rrg-aspuru

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"

cd /project/rrg-aspuru/memoozd/ts-tools

# Use 8 workers with 2 threads each on compute node (plenty of RAM)
export PHASE3_NWORKERS=8
export PHASE3_THREADS=2

bash run_phase3_300samples.sh

echo "Finished: $(date)"
