#!/bin/bash

#SBATCH --job-name=parallel_job
#SBATCH --mail-type=END, FAIL
#SBATCH --mail-user=ilgee@uchicago.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=pi-cdonnat
#SBATCH --mem=20G
#SBATCH --time=35:00:00
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
module load python
module load pytorch 

cd $SCRATCH/$USER/SelfGCon
echo $1
echo $2
python3 hyperparameter.py --result_file $SLURM_ARRAY_TASK_ID
~
~