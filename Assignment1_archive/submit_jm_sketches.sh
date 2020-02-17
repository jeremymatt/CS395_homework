#!/bin/bash
# specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=32
# Send Notification email
#SBATCH	--mail-type=FAIL,END
#SBATCH --mail-user=jmatt@uvm.edu
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=32G
# Maximum runtime of 10 minutes
#SBATCH --time=30:00:00
# Name of this job
#SBATCH --job-name=jm_arch
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=../output/%x_%j.out
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# your job execution follows:
time python ~/scratch/DeepLearning/github/CS395_homework/HW01_sketches_jmatt_driver.py


