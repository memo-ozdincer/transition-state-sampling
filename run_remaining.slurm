#!/bin/bash
#SBATCH --job-name=nr_gad_remaining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_remaining_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/nr_gad_remaining_%j.err
#SBATCH --account=rrg-aspuru

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"

# Run the remaining experiments script
bash /project/rrg-aspuru/memoozd/ts-tools/run_remaining.sh

echo "Finished: $(date)"
