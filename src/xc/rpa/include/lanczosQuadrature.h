#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#ifdef USE_MKL
#define MKL_Complex16 double _Complex
#include "mkl.h"
#include "mkl_lapacke.h"
#include "blacs.h"     // Cblacs_*
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#endif

#ifdef USE_SCALAPACK
#include <cblas.h>
#include <lapacke.h>
#include "blacs.h"     // Cblacs_*
#include "scalapack.h" // ScaLAPACK functions
#endif

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "restoreElectronicGroundState.h"
#include "collectOrbitals.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "tools_RPA.h"

//Haque TODO: Provide header comments
double  lanczos_quadrature(const SPARC_OBJ *pSPARC, RPA_OBJ *pRPA,
                            double* a, double* b,
                            int MAXIT, int unit_j, MPI_Comm comm,int omegaIndex);
double lanczos_quadrature_wrapper(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int n, int m, int omegaIndex);
void lanczos_quadrature_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex);