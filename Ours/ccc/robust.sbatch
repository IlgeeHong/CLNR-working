#!/bin/bash

#SBATCH --job-name=gpu-job
#SBATCH --output=logs/gpu_%A_%a.out
#SBATCH --error=logs/gpu_%A_%a.err
#SBATCH --time=35:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint=v100
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=1   # set this to the desired number of threads
#SBATCH --account=pi-cdonnat
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=END
#SBATCH --mail-user=ilgee@uchicago.edu

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
module load python
module load cuda
# source activate pytorch_env2  

cd $SCRATCH/$USER/SelfGCon
echo $1
echo $2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 Ours/ccc/robust.py
# conda deactivate
~
~
