#!/bin/sh

# SBATCH options:

#SBATCH --job-name=extract_cosmodc2_mm    # Job name
#SBATCH --output=log.log
#SBATCH --partition=htc               # Partition choice
#SBATCH --ntasks=1                    # Run a single task (by default tasks == CPU)
#SBATCH --mem=8000                    # Memory in MB per default
#SBATCH --time=0-5:00:00             # 7 days by default on htc partition


source /pbs/home/c/cpayerne/setup_mydesc.sh
python /pbs/throng/lsst/users/cpayerne/CLMassDC2/data/data_extraction/job/cosmodc2_weak_lensing_catalogs.py $START_NUM 

# $END_NUM

# python /pbs/throng/lsst/users/cpayerne/CLMassDC2/data/data_extraction/job/cosmodc2_weak_lensing_catalogs.py $DOWN $UP
