#!/bin/bash
#SBATCH --job-name=test-champ
#SBATCH --output=out.slurm
#SBATCH --error=err.slurm
#SBATCH --partition comp72
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00


module load python/anaconda-3.14
cd $SLURM_SUBMIT_DIR
cp *.yml /scratch/$SLURM_JOB_ID
cp ./champ/champ/run_yml_dir.py /scratch/$SLURM_JOB_ID
cd /scratch/$SLURM_JOB_ID
python run_yml_dir.py . 16
