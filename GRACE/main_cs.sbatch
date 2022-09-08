#!/bin/bash

#SBATCH --job-name=gpu-job
#SBATCH --output=logs/gpu_%A_%a.out
#SBATCH --error=logs/gpu_%A_%a.err
#SBATCH --time=35:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH --account=pi-cdonnat
#SBATCH --mem=20G
#SBATCH --mail-type=END
#SBATCH --mail-user=ilgee@uchicago.edu

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
module load python
module load cuda
source activate pytorch_env  

cd $SCRATCH/$USER/SelfGCon
echo $1
echo $2
python3 GRACE/main_cs.py
conda deactivate
~
~
