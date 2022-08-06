#!/bin/bash

#SBATCH --job-name=array-job
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --time=35:00:00
#SBATCH --account=pi-cdonnat
#SBATCH --ntasks=1
#SBATCH --partition=caslake
#SBATCH --mem=20G
#SBATCH --mail-type=END
#SBATCH --mail-user=ilgee@uchicago.edu

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
module load python
source activate pytorch_env  

cd $SCRATCH/$USER/SelfGCon
echo $1
echo $2
<<<<<<< HEAD
python3 Ours/main3.py
conda deactivate
=======
python3 Ours/main3.py --result_file $SLURM_ARRAY_TASK_ID
source deactivate
>>>>>>> 41b14f29454a4a24adc50f4e5eb7a30d7ffa4dac
~
~
