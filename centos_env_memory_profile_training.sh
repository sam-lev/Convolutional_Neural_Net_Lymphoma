#!/bin/bash
#SBATCH --time=41:06:66 # walltime, abbreviated by -t
#SBATCH --mem=70G
#SBATCH --job-name="setup"
#SBATCH -o mem_idx.out-%j # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e mem_idx.err-%j # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:2

source ~/.bashrc
conda activate fsm_tftpidx
#fileoutput="./memory_logs/mprofile-$(date +"%Y_%m_%d_%I_%M_%s%1N_%p").dat"
# --output "$fileoutput"
mprof run --multiprocess --include-children --output multiprocess_90_res ./fsm_training.sh
