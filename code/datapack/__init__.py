#!/home/sci/samlev/bin/bin/python3                                                              

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t                         
#SBATCH --mem=30G 
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:1

from .quality_random_crop import quality_random_crop 
from .lymphomaDataPack import lymphoma2
