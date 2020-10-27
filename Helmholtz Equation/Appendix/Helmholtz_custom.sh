#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=1500
#SBATCH --mail-type=end
#SBATCH --mail-user=p.bhustali@tu-braunschweig.de

module load cuda/10.0
module load lib/cudnn/7.6.1.34_cuda_10.0
module load anaconda/3-5.0.1

source activate tf2-gpu

srun python -u Helmholtz_Equation_custom.py