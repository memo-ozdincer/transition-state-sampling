#!/bin/bash
#SBATCH --job-name=reserve_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/reserve_node_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/reserve_node_%j.err
#SBATCH --account=rrg-aspuru

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

echo "Node reserved: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "Start: $(date)"
echo "Will hold until: $(date -d '+6 hours')"

sleep 21600

echo "Reservation expired at $(date)"
