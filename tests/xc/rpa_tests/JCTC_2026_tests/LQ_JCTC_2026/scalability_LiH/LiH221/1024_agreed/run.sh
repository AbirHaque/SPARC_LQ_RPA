#!/bin/bash
#SBATCH --account=gts-echow7-coda20                  # charge account
#SBATCH -N64 --ntasks-per-node=16                # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=2G                        # Memory per core
#SBATCH --time=5:00:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                              # QOS Name
#SBATCH -o3_23_2026_%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=abirhaque@gatech.edu        # E-mail address for notifications

module load gcc/12.3.0
module load openmpi/4.1.5
module load netlib-scalapack/2.2.0-ompi
module load openblas
module load fftw/3.3.10-ompi

#srun -n 2048 ../../../../../lib/sparc_2 -name LiH_rpa_gamma
#srun -n 512 ../../../../../lib/sparc_3_no_interp_fast -name LiH_rpa_gamma
#srun -n 1024  ../../../../../../../lib/sparc_3_no_interp_fast -name LiH
srun -n 1024  ../../../../../../../../lib/sparc -name LiH
