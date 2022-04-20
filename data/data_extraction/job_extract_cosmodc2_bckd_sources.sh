#!/usr/bin/bash

#SLURM options:

#SBATCH --job-name=computemodel    # Job name
#SBATCH --output=log.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=1                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=3000                    # Memory in MB per default
#SBATCH --time=1-00:00:00             # 7 days by default on htc partition
#SBATCH --array=100

N_JOB=$SLURM_ARRAY_TASK_MAX
N_POINTS=10000
N_PER_JOB=$(($N_POINTS / $N_JOB))

START_NUM=$(( ($SLURM_ARRAY_TASK_ID-1) * $N_PER_JOB ))
END_NUM=$(( ($SLURM_ARRAY_TASK_ID) * $N_PER_JOB))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs from $START_NUM to $END_NUM

source /pbs/home/c/cpayerne/setup_mydesc.sh
python /pbs/throng/lsst/users/cpayerne/ClusterLikelihoods/modules/pinocchio_analysis/compute_model.py $N_Z_BIN $N_M_BIN $START_NUM $END_NUM
