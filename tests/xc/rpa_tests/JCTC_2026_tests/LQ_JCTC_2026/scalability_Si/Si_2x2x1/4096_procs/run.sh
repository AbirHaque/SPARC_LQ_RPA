#!/bin/bash
#SBATCH --account=gts-echow7-coda20                  # charge account
#SBATCH -N256 --ntasks-per-node=16                # Number of nodes and cores per node re>
#SBATCH --mem-per-cpu=2G                        # Memory per core
#SBATCH --time=4:00:00                                    # Duration of the job (Ex: 1>
#SBATCH -qinferno                               # QOS Name
#SBATCH -o7_8_2026_%j.out                         # Combined output and error messages>
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=abirhaque@gatech.edu        # E-mail address for notifications

module load gcc/12.3.0
module load openmpi/4.1.5
module load netlib-scalapack/2.2.0-ompi
module load openblas
module load fftw/3.3.10-ompi

srun -n 4096 ../../../../../../../../lib/sparc -name Si
