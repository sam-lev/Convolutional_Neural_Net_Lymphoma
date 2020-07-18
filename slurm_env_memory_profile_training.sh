#!/bin/bash
#SBATCH --time=96:06:66 # walltime, abbreviated by -t
#SBATCH --mem=120G
#SBATCH --job-name="resoOptNN"
#SBATCH -o res90_new.out-%j # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e res90_new.err-%j # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate slurmtftpidx
fileoutput="res90-mprofile-$(date +"%Y_%m_%d_%I_%M_%s%1N_%p").dat"
# --output "$fileoutput"
mprof run --multiprocess --include-children --output $fileoutput ./medusa_training.sh
