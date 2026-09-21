/**
 * @file    linearSolvers.c
 * @brief   This file contains the linear solvers for solving Sternheimer equations
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#ifdef USE_MKL
#define MKL_Complex16 double _Complex
#define MKL_INT int
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

#include "linearSolvers.h"
#include "tools_RPA.h"

int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double *, double, double, int, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int rhsBlockSize,
     int flagPreconditioner, void (*precond)(SPARC_OBJ *, double *, double *, int, double *, double), double *poisC_invE_precond, double Icoeff,
     double RHS_frobnormsq, double sternRelativeResTol, int maxIter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime) {
    double lhsTimeRecord = 0.0; double solveMuTimeRecord = 0.0; double multipleTimeRecord = 0.0;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int rhsLength = DMnd * rhsBlockSize;
    int rhoSize = rhsBlockSize * rhsBlockSize;

    double _Complex *LHSx = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
    double _Complex *V = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
    double *VReal, *WReal, *VImag, *WImag;
    #ifdef DEBUG
    double t1 = MPI_Wtime();
    #endif
    lhsfun(pSPARC, spn_i, psi, epsilon, omega, flagPQ, deltaPsisReal, deltaPsisImag, LHSx, rhsBlockSize);
    #ifdef DEBUG
    double t2 = MPI_Wtime();
    lhsTimeRecord += (t2 - t1);
    #endif
    for (int rhsIndex = 0; rhsIndex < rhsLength; rhsIndex++) {
        V[rhsIndex] = SternheimerRhs[rhsIndex] - LHSx[rhsIndex];
    }
    for (int vecIndex = 0; vecIndex < rhsBlockSize; vecIndex++) {
        Vector2Norm_complex(V + vecIndex*DMnd, DMnd, &(resNormRecords[vecIndex]), pSPARC->dmcomm);
    }
    double _Complex *W = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
    if (flagPreconditioner) {
        VReal = (double*)calloc(sizeof(double), rhsLength);
        VImag = (double*)calloc(sizeof(double), rhsLength);
        WReal = (double*)calloc(sizeof(double), rhsLength);
        WImag = (double*)calloc(sizeof(double), rhsLength);
        divide_complex_vectors(V, VReal, VImag, rhsLength);
        precond(pSPARC, VReal, WReal, rhsBlockSize, poisC_invE_precond, Icoeff);
        precond(pSPARC, VImag, WImag, rhsBlockSize, poisC_invE_precond, Icoeff);
        for (int i = 0; i < rhsLength; i++) {
            W[i] = WReal[i] + WImag[i]*I;
        }
    } else {
        memcpy(W, V, sizeof(double _Complex) * rhsLength);
    }
    double _Complex *rho = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
    double _Complex *lapackRho = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
    // Reminder: if there is domain parallelization, then it is necessary to call pdgemm_ function in ScaLapack with blacs
    // to make the distributed matrix multiplication
    double _Complex one = 1.0; double _Complex zero = 0.0; double _Complex minus1 = -1.0;
    #ifdef DEBUG
    double t3 = MPI_Wtime();
    #endif
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, rhsBlockSize, rhsBlockSize, DMnd,
                  &one, V, DMnd,
                  W, DMnd, &zero,
                  rho, rhsBlockSize);
    #ifdef DEBUG
    double t4 = MPI_Wtime();
    multipleTimeRecord += (t4 - t3);
    #endif
    double _Complex *P = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
    double _Complex *beta = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
    double _Complex *U = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
    double _Complex *mu = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
    double _Complex *lapackMu = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
    double _Complex *alpha = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
    double _Complex *rhoNew = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
    double _Complex *midVariable = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
    double *dividedReal = (double*)calloc(sizeof(double), rhsLength);
    double *dividedImag = (double*)calloc(sizeof(double), rhsLength);
    int info;
    int ix;
    for (ix = 0; ix < maxIter; ix++) {
        if (judge_converge(ix, rhsBlockSize, RHS_frobnormsq, sternRelativeResTol, resNormRecords)) {
            break;
        }
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize,
                  &one, P, DMnd,
                  beta, rhsBlockSize, &zero,
                  midVariable, DMnd);
        for (int i = 0; i < rhsLength; i++) {
            P[i] = W[i] + midVariable[i]; // P = W + P*beta;
        }
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        divide_complex_vectors(P, dividedReal, dividedImag, rhsLength);
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        lhsfun(pSPARC, spn_i, psi, epsilon, omega, flagPQ, dividedReal, dividedImag, U, rhsBlockSize); // U = Afun(P);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        lhsTimeRecord += (t2 - t1);
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, rhsBlockSize, rhsBlockSize, DMnd,
                  &one, U, DMnd,
                  P, DMnd, &zero,
                  mu, rhsBlockSize); // mu = U.' * P;
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        int ipiv[rhsBlockSize];
        memcpy(alpha, rho, sizeof(double _Complex)*rhoSize);
        memcpy(lapackMu, mu, sizeof(double _Complex)*rhoSize);
        #ifdef DEBUG
        double t5 = MPI_Wtime();
        #endif
        info = LAPACKE_zsysv( LAPACK_COL_MAJOR, 'L', rhsBlockSize, rhsBlockSize, lapackMu, rhsBlockSize, ipiv, alpha, rhsBlockSize ); // alpha = mu \ rho;
        #ifdef DEBUG
        double t6 = MPI_Wtime();
        solveMuTimeRecord += (t6 - t5);
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize,
                  &one, P, DMnd,
                  alpha, rhsBlockSize, &zero,
                  midVariable, DMnd);
        for (int i = 0; i < rhsLength; i++) {
            deltaPsisReal[i] += creal(midVariable[i]);
            deltaPsisImag[i] += cimag(midVariable[i]); // X = X + P*alpha;
        }
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize,
                  &minus1, U, DMnd,
                  alpha, rhsBlockSize, &one,
                  V, DMnd); // V = V - U*alpha;
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        if (flagPreconditioner) {
            divide_complex_vectors(V, VReal, VImag, rhsLength);
            precond(pSPARC, VReal, WReal, rhsBlockSize, poisC_invE_precond, Icoeff);
            precond(pSPARC, VImag, WImag, rhsBlockSize, poisC_invE_precond, Icoeff);
            for (int i = 0; i < rhsLength; i++) {
                W[i] = WReal[i] + WImag[i]*I;
            }
        } else {
            memcpy(W, V, sizeof(double _Complex) * rhsLength);
        }
        for (int vecIndex = 0; vecIndex < rhsBlockSize; vecIndex++) {
            Vector2Norm_complex(V + vecIndex*DMnd, DMnd, &(resNormRecords[(ix + 1)*rhsBlockSize + vecIndex]), pSPARC->dmcomm);
        }
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, rhsBlockSize, rhsBlockSize, DMnd,
                  &one, V, DMnd,
                  W, DMnd, &zero,
                  rhoNew, rhsBlockSize); // rho_new = V.' * W;
        #ifdef DEBUG
        t4 = MPI_Wtime();
        multipleTimeRecord += (t4 - t3);
        #endif
        memcpy(beta, rhoNew, sizeof(double _Complex)*rhoSize);
        memcpy(lapackRho, rho, sizeof(double _Complex)*rhoSize);
        #ifdef DEBUG
        t5 = MPI_Wtime();
        #endif
        info = LAPACKE_zsysv( LAPACK_COL_MAJOR, 'L', rhsBlockSize, rhsBlockSize, lapackRho, rhsBlockSize, ipiv, beta, rhsBlockSize ); // beta = rho \ rho_new;
        #ifdef DEBUG
        t6 = MPI_Wtime();
        solveMuTimeRecord += (t6 - t5);
        #endif
        memcpy(rho, rhoNew, sizeof(double _Complex)*rhoSize); // rho = rho_new;
    }
    *lhsTime = lhsTimeRecord; *solveMuTime = solveMuTimeRecord; *multipleTime = multipleTimeRecord;

    free(LHSx);
    free(V);
    free(W);
    free(rho);
    free(lapackRho);
    free(P);
    free(beta);
    free(U);
    free(mu);
    free(lapackMu);
    free(alpha);
    free(rhoNew);
    free(midVariable);
    free(dividedReal);
    free(dividedImag);
    if (flagPreconditioner) {
        free(VReal); free(WReal); free(VImag); free(WImag);
    }
    return ix;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include <mpi.h>

typedef struct {
    int is_initialized;
    int DMnd;
    double theta_min; 
    double theta_max; 
} A0_Spectrum_Cache;

static A0_Spectrum_Cache a0_cache = { .is_initialized = 0 };

void free_sternheimer_condition_number_cache() {
    a0_cache.is_initialized = 0;
}

double estimate_sternheimer_condition_number_k2(
    void (*lhsfun)(SPARC_OBJ*, int, double *, double, double, int, double *, double *, double _Complex*, int),
    SPARC_OBJ *pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ,
    int lanczos_steps
) {
    int DMnd = pSPARC->Nd_d_dmcomm;

    if (!a0_cache.is_initialized || a0_cache.DMnd != DMnd) {
        double *alphas = (double *)malloc(sizeof(double) * lanczos_steps);
        double *betas  = (double *)malloc(sizeof(double) * (lanczos_steps - 1));
        
        double *q_prev = (double *)calloc(DMnd, sizeof(double));
        double *q_curr = (double *)malloc(sizeof(double) * DMnd);
        double *w_real = (double *)malloc(sizeof(double) * DMnd);
        double *u_imag = (double *)calloc(DMnd, sizeof(double));
        double _Complex *lhs_out = (double _Complex *)malloc(sizeof(double _Complex) * DMnd);

        double norm_v = sqrt((double)DMnd);
        for (int i = 0; i < DMnd; i++) q_curr[i] = 1.0 / norm_v;

        int k_eff = lanczos_steps;

        for (int j = 0; j < lanczos_steps; j++) {
            lhsfun(pSPARC, spn_i, psi, 0.0, 0.0, flagPQ, q_curr, u_imag, lhs_out, 1);

            for (int i = 0; i < DMnd; i++) w_real[i] = creal(lhs_out[i]);

            double alpha = 0.0;
            for (int i = 0; i < DMnd; i++) alpha += q_curr[i] * w_real[i];
            alphas[j] = alpha;

            for (int i = 0; i < DMnd; i++) {
                w_real[i] -= alphas[j] * q_curr[i] + ((j > 0) ? betas[j - 1] * q_prev[i] : 0.0);
            }

            if (j < lanczos_steps - 1) {
                double w_norm_sq = 0.0;
                for (int i = 0; i < DMnd; i++) w_norm_sq += w_real[i] * w_real[i];
                double beta = sqrt(w_norm_sq);

                if (beta < 1e-12) { k_eff = j + 1; break; }

                betas[j] = beta;
                memcpy(q_prev, q_curr, sizeof(double) * DMnd);
                for (int i = 0; i < DMnd; i++) q_curr[i] = w_real[i] / beta;
            }
        }

        double *d = (double *)malloc(sizeof(double) * k_eff);
        double *e = (double *)malloc(sizeof(double) * (k_eff - 1));
        memcpy(d, alphas, sizeof(double) * k_eff);
        if (k_eff > 1) memcpy(e, betas, sizeof(double) * (k_eff - 1));

        LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', k_eff, d, e, NULL, k_eff);

        a0_cache.theta_min = d[0];
        a0_cache.theta_max = d[k_eff - 1];
        a0_cache.DMnd = DMnd;
        a0_cache.is_initialized = 1;

        free(alphas); free(betas); free(q_prev); free(q_curr);
        free(w_real); free(u_imag); free(lhs_out); free(d); free(e);
    }

    double omega_sq = omega * omega;
    double mod_sq_min1 = (a0_cache.theta_min - epsilon) * (a0_cache.theta_min - epsilon) + omega_sq;
    double mod_sq_max1 = (a0_cache.theta_max - epsilon) * (a0_cache.theta_max - epsilon) + omega_sq;

    double min_mod_sq;
    if (epsilon >= a0_cache.theta_min && epsilon <= a0_cache.theta_max) {
        min_mod_sq = omega_sq;
    } else {
        min_mod_sq = fmin(mod_sq_min1, mod_sq_max1);
    }

    double max_mod_sq = fmax(mod_sq_min1, mod_sq_max1);
    if (min_mod_sq < 1e-16) min_mod_sq = 1e-16;

    return sqrt(max_mod_sq / min_mod_sq);
}

int judge_converge(int ix, int numVecs, const double RHS_frobnormsq, double sternRelativeResTol, const double *resNormRecords) {
    // int judge = 1;
    double resid_frobnormsq = 0;
    for (int col = 0; col < numVecs; col++)
    {
        resid_frobnormsq += resNormRecords[numVecs*ix + col]*resNormRecords[numVecs*ix + col];
    }
    int judge = (resid_frobnormsq / RHS_frobnormsq) > sternRelativeResTol*sternRelativeResTol ? 0 : 1;
    // double sum = 0.0;
    // for (int i = 0; i < numVecs; i++) {
    //     sum += resNormRecords[ix*numVecs + i];
    // }
    // int judge = (sum / sqrt((double)numVecs)) > tol ? 0 : 1;
    return judge;
}


int block_CRM(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, int kPq, double _Complex *psi_kpt, double epsilon, double omega, int flagPQ, double _Complex *deltaPsisPlus, double _Complex *deltaPsisMinus,
     double _Complex *SternheimerRhs, int rhsBlockSize,
     // int flagPreconditioner, void (*precond)(SPARC_OBJ *, double *, double *, int, double *, double), double *poisC_invE_precond, double Icoeff,
     double RHS_frobnormsq, double sternRelativeResTol, int maxInnerIter, int maxOuterIter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime) {
    double lhsTimeRecord = 0.0; double solveMuTimeRecord = 0.0; double multipleTimeRecord = 0.0;
    int DMnd = pSPARC->Nd_d_dmcomm;
    int rhsLength = DMnd * rhsBlockSize;
    int rhoSize = rhsBlockSize * rhsBlockSize;

    double _Complex omegaDiff = -2 * I *omega;
    double _Complex ONE = 1.0;
    double _Complex ZERO = 0.0;

    int outerIter = 0, innerIter = 0;
    int outerIterFlag = 1;
    while (outerIterFlag) {
        double _Complex *Vx = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
        double _Complex *Vy = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
        #ifdef DEBUG
        double t1 = MPI_Wtime();
        #endif
        lhsfun(pSPARC, spn_i, kPq, epsilon, omega, deltaPsisPlus, Vx, rhsBlockSize);
        lhsfun(pSPARC, spn_i, kPq, epsilon, -omega, deltaPsisMinus, Vy, rhsBlockSize);
        #ifdef DEBUG
        double t2 = MPI_Wtime();
        lhsTimeRecord += (t2 - t1);
        #endif
        for (int i = 0; i < rhsLength; i++) {
            Vx[i] = SternheimerRhs[i] - Vx[i]; // B - Afun_pls(X);
            Vy[i] = SternheimerRhs[i] - Vy[i];
        }
        int initialTotalIter = outerIter*(maxInnerIter + 1);
        for (int vecIndex = 0; vecIndex < rhsBlockSize; vecIndex++) {
            double theVecNorm;
            Vector2Norm_complex(Vx + vecIndex*DMnd, DMnd, &theVecNorm, pSPARC->dmcomm);
            resNormRecords[initialTotalIter*2] += theVecNorm*theVecNorm;
            Vector2Norm_complex(Vy + vecIndex*DMnd, DMnd, &theVecNorm, pSPARC->dmcomm);
            resNormRecords[initialTotalIter*2 + 1] += theVecNorm*theVecNorm;
        }
        resNormRecords[initialTotalIter*2] /= RHS_frobnormsq;
        resNormRecords[initialTotalIter*2 + 1] /= RHS_frobnormsq;
    
        double _Complex *Px = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *Py = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *Ux = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *Uy = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        if (outerIter % 2) {
            memcpy(Px, Vy, sizeof(double _Complex) * rhsLength);
            memcpy(Py, Vy, sizeof(double _Complex) * rhsLength); // initial Py should be identical to Vx
        } else {
            memcpy(Px, Vx, sizeof(double _Complex) * rhsLength);
            memcpy(Py, Vx, sizeof(double _Complex) * rhsLength); // initial Py should be identical to Vx
        }
        #ifdef DEBUG
        t1 = MPI_Wtime();
        #endif
        lhsfun(pSPARC, spn_i, kPq, epsilon, omega, Px, Ux, rhsBlockSize); // Ux = Afun_pls(Px);
        #ifdef DEBUG
        t2 = MPI_Wtime();
        lhsTimeRecord += (t2 - t1);
        #endif
        for (int i = 0; i < rhsLength; i++) {
            Uy[i] = Ux[i] + omegaDiff*Px[i];
        }
        double _Complex *Px_old = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
        double _Complex *Ux_old = (double _Complex*)calloc(sizeof(double _Complex), rhsLength);
        double _Complex *rho_old = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
        double _Complex *rho = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
        double _Complex *rhoCopy = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
        for (int row = 0; row < rhsBlockSize; row++) {
            rho_old[rhsBlockSize*row + row] = 1.0;
        }
    
        double _Complex *alphax = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
        double _Complex *mu = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
        double _Complex *alphay = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
        double _Complex *beta = (double _Complex*)calloc(sizeof(double _Complex), rhoSize);
        double _Complex *midVar1 = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *midVar2 = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *W = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
    
        double _Complex *Px_new = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *Py_new = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *Ux_new = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        double _Complex *Uy_new = (double _Complex*)malloc(sizeof(double _Complex) * rhsLength);
        
        innerIter = 0;
        double tolSquare = sternRelativeResTol * sternRelativeResTol;
        for (innerIter = 0; innerIter < maxInnerIter; innerIter++) {
            int lastTotalIter = initialTotalIter + innerIter;
            if ((resNormRecords[lastTotalIter*2] < tolSquare) && (resNormRecords[lastTotalIter*2 + 1] < tolSquare)) {
                outerIterFlag = 0;
                break;
            }
    
            #ifdef DEBUG
            double t3 = MPI_Wtime();
            #endif
            for (int i = 0; i < rhsLength; i++) {
                midVar1[i] = conj(Ux[i]);
                midVar2[i] = conj(Uy[i]);
            }
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, rhsBlockSize, rhsBlockSize, DMnd, &ONE, midVar1, DMnd, Ux, DMnd, &ZERO, rho, rhsBlockSize); // rho = Ux'*Ux;
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, rhsBlockSize, rhsBlockSize, DMnd, &ONE, midVar1, DMnd, Vx, DMnd, &ZERO, alphax, rhsBlockSize); // Ux'*Vx
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, rhsBlockSize, rhsBlockSize, DMnd, &ONE, midVar2, DMnd, Vy, DMnd, &ZERO, alphay, rhsBlockSize); // Uy'*Vy
            #ifdef DEBUG
            double t4 = MPI_Wtime();
            multipleTimeRecord += (t4 - t3);
            double t5 = MPI_Wtime();
            #endif
            memcpy(rhoCopy, rho, sizeof(double _Complex) * rhoSize);
            int info = LAPACKE_zposv(LAPACK_COL_MAJOR, 'U', rhsBlockSize, rhsBlockSize, rhoCopy, rhsBlockSize, alphax, rhsBlockSize); // alphax = rho \ (Ux'*Vx);
            memcpy(rhoCopy, rho, sizeof(double _Complex) * rhoSize);
            info = LAPACKE_zposv(LAPACK_COL_MAJOR, 'U', rhsBlockSize, rhsBlockSize, rhoCopy, rhsBlockSize, alphay, rhsBlockSize); // alphay = rho \ (Uy'*Vy);
            #ifdef DEBUG
            double t6 = MPI_Wtime();
            solveMuTimeRecord += (t6 - t5);
            t3 = MPI_Wtime();
            #endif
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Px, DMnd, alphax, rhsBlockSize, &ZERO, midVar1, DMnd);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Ux, DMnd, alphax, rhsBlockSize, &ZERO, midVar2, DMnd);
            for (int index = 0; index < rhsLength; index ++) {
                deltaPsisPlus[index] += midVar1[index]; // X = X + Px*alphax;
                Vx[index] -= midVar2[index]; // Vx = Vx - Ux*alphax;
            }
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Py, DMnd, alphay, rhsBlockSize, &ZERO, midVar1, DMnd);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Uy, DMnd, alphay, rhsBlockSize, &ZERO, midVar2, DMnd);
            for (int index = 0; index < rhsLength; index ++) {
                deltaPsisMinus[index] += midVar1[index]; // Y = Y + Py*alphay;
                Vy[index] -= midVar2[index]; // Vy = Vy - Uy*alphay;
            }
            #ifdef DEBUG
            t4 = MPI_Wtime();
            multipleTimeRecord += (t4 - t3);
            #endif
    
            for (int vecIndex = 0; vecIndex < rhsBlockSize; vecIndex++) {
                double theVecNorm;
                Vector2Norm_complex(Vx + vecIndex*DMnd, DMnd, &theVecNorm, pSPARC->dmcomm);
                resNormRecords[lastTotalIter*2 + 2] += theVecNorm*theVecNorm;
                Vector2Norm_complex(Vy + vecIndex*DMnd, DMnd, &theVecNorm, pSPARC->dmcomm);
                resNormRecords[lastTotalIter*2 + 3] += theVecNorm*theVecNorm;
            }
            resNormRecords[lastTotalIter*2 + 2] /= RHS_frobnormsq;
            resNormRecords[lastTotalIter*2 + 3] /= RHS_frobnormsq;
    
            #ifdef DEBUG
            t1 = MPI_Wtime();
            #endif
            lhsfun(pSPARC, spn_i, kPq, epsilon, 0, Ux, W, rhsBlockSize); // W = Afun(Ux);
            #ifdef DEBUG
            t2 = MPI_Wtime();
            lhsTimeRecord += (t2 - t1);
            t3 = MPI_Wtime();
            #endif
            for (int i = 0; i < rhsLength; i++) {
                midVar1[i] = conj(Ux[i]);
            }
            cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, rhsBlockSize, rhsBlockSize, DMnd, &ONE, midVar1, DMnd, W, DMnd, &ZERO, mu, rhsBlockSize); // Ux'*W
            #ifdef DEBUG
            t4 = MPI_Wtime();
            multipleTimeRecord += (t4 - t3);
            t5 = MPI_Wtime();
            #endif
            memcpy(rhoCopy, rho, sizeof(double _Complex) * rhoSize);
            info = LAPACKE_zposv(LAPACK_COL_MAJOR, 'U', rhsBlockSize, rhsBlockSize, rhoCopy, rhsBlockSize, mu, rhsBlockSize); // mu = rho \ (Ux'*W);
            memcpy(beta, rho, sizeof(double _Complex) * rhoSize);
            memcpy(rhoCopy, rho_old, sizeof(double _Complex) * rhoSize);
            info = LAPACKE_zposv(LAPACK_COL_MAJOR, 'U', rhsBlockSize, rhsBlockSize, rhoCopy, rhsBlockSize, beta, rhsBlockSize); // beta = rho_old \ rho;
            #ifdef DEBUG
            t6 = MPI_Wtime();
            solveMuTimeRecord += (t6 - t5);
            t3 = MPI_Wtime();
            #endif
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Ux, DMnd, mu, rhsBlockSize, &ZERO, midVar1, DMnd);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Ux_old, DMnd, beta, rhsBlockSize, &ZERO, midVar2, DMnd);
            for (int index = 0; index < rhsLength; index++) { // Ux_new = W - Ux*mu - Ux_old*beta;
                Ux_new[index] = W[index] - midVar1[index] - midVar2[index];
            }
    
            for (int row = 0; row < rhsBlockSize; row++) {
                mu[row + row * rhsBlockSize] += I * omega;
            }
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Px, DMnd, mu, rhsBlockSize, &ZERO, midVar1, DMnd);
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, rhsBlockSize, rhsBlockSize, &ONE, Px_old, DMnd, beta, rhsBlockSize, &ZERO, midVar2, DMnd);
            for (int index = 0; index < rhsLength; index++) { // Px_new = Ux - Px*(mu + sigma*eye(blksz)) - Px_old*beta;
                Px_new[index] = Ux[index] - midVar1[index] - midVar2[index];
            }
    
            memcpy(Py_new, Px_new, sizeof(double _Complex) * rhsLength); // Py_new = Px_new;
            for (int index = 0; index < rhsLength; index++) { // Uy_new = Ux_new + mat_diff*Px_new;
                Uy_new[index] = Ux_new[index] + omegaDiff * Px_new[index];
            }
            #ifdef DEBUG
            t4 = MPI_Wtime();
            multipleTimeRecord += (t4 - t3);
            #endif
            memcpy(Px_old, Px, sizeof(double _Complex) * rhsLength); // Px_old = Px;
            memcpy(Px, Px_new, sizeof(double _Complex) * rhsLength); // Px = Px_new;
            memcpy(Py, Py_new, sizeof(double _Complex) * rhsLength); // Py = Py_new;
    
            memcpy(Ux_old, Ux, sizeof(double _Complex) * rhsLength); // Ux_old = Ux;
            memcpy(Ux, Ux_new, sizeof(double _Complex) * rhsLength); // Ux = Ux_new;
            memcpy(Uy, Uy_new, sizeof(double _Complex) * rhsLength); // Uy = Uy_new;
    
            memcpy(rho_old, rho, sizeof(double _Complex) * rhoSize); // rho_old = rho;
    
            if (innerIter == maxInnerIter - 1) {
                printf("outerIter %d terminated. resNormRecords[%d][%d] %9.6f %9.6f, resNormRecords[%d][%d] %9.6f %9.6f\n",
                    outerIter, initialTotalIter*2, initialTotalIter*2 + 1, resNormRecords[initialTotalIter*2], resNormRecords[initialTotalIter*2 + 1],
                    (initialTotalIter + maxInnerIter)*2, (initialTotalIter + maxInnerIter)*2 + 1, resNormRecords[(initialTotalIter + maxInnerIter)*2], resNormRecords[(initialTotalIter + maxInnerIter)*2 + 1]);
            }
        }

        #ifdef DEBUG
        printf("kPq %d, %d outerIter, %d innerIter\n", kPq, outerIter, innerIter);
        #endif
        if ((outerIter == maxOuterIter - 1) && (innerIter == maxInnerIter)) {
            printf("Block CRM terminated without converging to the desired tolerance.\n");
        }
        *lhsTime = lhsTimeRecord; *solveMuTime = solveMuTimeRecord; *multipleTime = multipleTimeRecord;
        
        free(Vx);
        free(Vy);
        free(Px);
        free(Py);
        free(Ux);
        free(Uy);
        free(Px_old);
        free(Ux_old);
        free(rho_old);
        free(rho);
        free(rhoCopy);
        free(alphax);
        free(mu);
        free(alphay);
        free(beta);
        free(midVar1);
        free(midVar2);
        free(W);
        free(Px_new);
        free(Py_new);
        free(Ux_new);
        free(Uy_new);
        outerIter++;
        if (outerIter == maxOuterIter) {
            outerIterFlag = 0;
        }
    }

    return (outerIter*(maxInnerIter + 1) + innerIter);
}

int AAR_sternheimer_kpt(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
    SPARC_OBJ* pSPARC, int spn_i, int kPq, double _Complex *psi_kpt, double epsilon, double omega, int flagPQ, double _Complex *x,
    double _Complex *SternheimerRhs,
    // void (*precond_fun)(SPARC_OBJ *,int,double,double *,double*,MPI_Comm), double c, 
    double RHS_frobnormsq, double tol, int max_iter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime,
    double omegaR, double betaA, int m, int p)
{
    
    int i, iter_count, i_hist;
    double _Complex *r, *x_old, *f, *f_old, *X, *F;
    double b_2norm, r_2norm;

#ifdef DEBUG
    double t1, t2;
#endif
    double tt1, tt2, ttot, ttot2;
    
    ttot = ttot2 = 0.0;
    
    int DMnd = pSPARC->Nd_d_dmcomm;
    // allocate memory for storing x, residual and preconditioned residual in the local domain
    r       = (double _Complex*)malloc( DMnd * sizeof(double _Complex) );  // residual vector, r = b - Ax
    x_old   = (double _Complex*)malloc( DMnd * sizeof(double _Complex) );
    f       = (double _Complex*)malloc( DMnd * sizeof(double _Complex) ); // preconditioned residual vector, f = inv(M) * r
    f_old   = (double _Complex*)malloc( DMnd * sizeof(double _Complex) );    
    assert(r != NULL && x_old != NULL && f != NULL && f_old != NULL);

    // allocate memory for storing X, F history matrices
    X = (double _Complex *)calloc( DMnd * m, sizeof(double _Complex) ); 
    F = (double _Complex *)calloc( DMnd * m, sizeof(double _Complex) ); 
    assert(X != NULL && F != NULL);

    // initialize x_old as x0 (initial guess vector)
    for (i = 0; i < DMnd; i++) 
        x_old[i] = x[i];

    Vector2Norm_complex(SternheimerRhs, DMnd, &b_2norm, pSPARC->dmcomm); // b_2norm = ||b||
    // find initial residual vector r = b - Ax, and its 2-norm
    lhsfun(pSPARC, spn_i, kPq, epsilon, omega, x, r, 1);
    for (int i = 0; i < DMnd; i++) {
        r[i] = SternheimerRhs[i] - r[i];
    }
    Vector2Norm_complex(r, DMnd, &r_2norm, pSPARC->dmcomm); // r_2norm = ||r||
    resNormRecords[0] = r_2norm * r_2norm / RHS_frobnormsq;

    // replace the abs tol by scaled tol: tol * ||b||
    tol *= b_2norm; 
    r_2norm = tol + 1.0; // to reduce communication
    iter_count = 0;
    while (r_2norm > tol && iter_count < max_iter) {  
        // *** calculate preconditioned residual f *** //
        // precond_fun(pSPARC, N, c, r, f, comm); // f = inv(M) * r     
        memcpy(f, r, DMnd * sizeof(double _Complex));
        // *** store residual & iteration history *** //
        if (iter_count > 0) {
            i_hist = (iter_count - 1) % m;
            for (i = 0; i < DMnd; i++) {
                X[i_hist*DMnd + i] = x[i] - x_old[i];
                F[i_hist*DMnd + i] = f[i] - f_old[i];
            }
        }
        
        for (i = 0; i < DMnd; i++) {
            x_old[i] = x[i];
            f_old[i] = f[i];
        }
        
        if((iter_count+1) % p == 0 && iter_count > 0) {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/
            tt1 = MPI_Wtime();
            // AndersonExtrapolation(N, m, x, x, f, X, F, beta, comm);
            AndersonExtrapolation_complex(DMnd, m, x, x_old, f, X, F, (double _Complex)betaA, pSPARC->dmcomm);
            tt2 = MPI_Wtime();
            ttot += (tt2-tt1);

            tt1 = MPI_Wtime();
            // update residual r = b - Ax
            lhsfun(pSPARC, spn_i, kPq, epsilon, omega, x, r, 1);
            for (int i = 0; i < DMnd; i++) {
                r[i] = SternheimerRhs[i] - r[i];
            }
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            Vector2Norm_complex(r, DMnd, &r_2norm, pSPARC->dmcomm); // r_2norm = ||r||
            resNormRecords[iter_count + 1] = r_2norm * r_2norm / RHS_frobnormsq;
        } else {
            /***********************
             *  Richardson update  *
             ***********************/
            for (i = 0; i < DMnd; i++)
                // x[i] = x[i] + omega * f[i];  
                x[i] = x_old[i] + omegaR * f[i];  
            tt1 = MPI_Wtime();   
            // update residual r = b - Ax
            lhsfun(pSPARC, spn_i, kPq, epsilon, omega, x, r, 1);
            for (int i = 0; i < DMnd; i++) {
                r[i] = SternheimerRhs[i] - r[i];
            }
            tt2 = MPI_Wtime();
            ttot2 += (tt2 - tt1);
            Vector2Norm_complex(r, DMnd, &r_2norm, pSPARC->dmcomm); // r_2norm = ||r||
            resNormRecords[iter_count + 1] = r_2norm * r_2norm / RHS_frobnormsq;
        }
        iter_count++;
    }
  
    *lhsTime = ttot2; *solveMuTime = ttot;
    // deallocate memory
    free(x_old);
    free(f_old);
    free(f);
    free(r);
    free(X);
    free(F);

    return iter_count;
}

// x_new = x_prev + beta*res - (DX + beta*DF)*(pinv(DF'*DF)*(DF'*res));
void  AndersonExtrapolation_complex(
        const int N, const int m, double _Complex *x_kp1, const double _Complex *x_k, 
        const double _Complex *f_k, const double _Complex *X, const double _Complex *F, 
        const double _Complex betaA, MPI_Comm comm) 
{
    unsigned i;
    double _Complex *f_wavg = (double _Complex*)malloc( N * sizeof(double _Complex) );
    
    // find the weighted average vectors
    // use Nspden = 0, opt = 0 here. default as previous
    
    // find extrapolation weigths Gamma = inv(F^T * F) * F^T * f_k
    double _Complex minusOne = -1.0;
    double _Complex one = 1.0;
    double _Complex zero = 0.0;
    int matrank;
    double _Complex *FtF = (double _Complex*)malloc( m * m * sizeof(double _Complex) );
    double *s   = (double*)malloc( m * sizeof(double) );
    double _Complex *Gamma = (double _Complex*)calloc( m , sizeof(double _Complex) );
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, m, N, &one, F, 
        N, F, N, &zero, FtF, m);
    cblas_zgemv(CblasColMajor, CblasTrans, N, m, &one, F, 
        N, f_k, 1, &zero, Gamma, 1);
    // Sum the local results of F^T * F and F^T * f (GLOBAL)
    MPI_Allreduce(MPI_IN_PLACE, FtF, m*m, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, Gamma, m, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    LAPACKE_zgelsd(LAPACK_COL_MAJOR, m, m, 1, FtF, m, Gamma, m, s, -1.0, &matrank);

    free(FtF);
    free(s);
    
    // find weighted average x_{k+1} = x_k - X*Gamma
    // memcpy(x_wavg, x_k, N*sizeof(*x_k)); // copy x_k into x_wavg

    for (i = 0; i < N; i++) x_kp1[i] = x_k[i];
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, m, &minusOne, X, 
        N, Gamma, 1, &one, x_kp1, 1);

    // find weighted average f_{k+1} = f_k - F*Gamma
    // memcpy(f_wavg, f_k, N*sizeof(*f_k)); // copy f_k into f_wavg
    for (i = 0; i < N; i++) f_wavg[i] = f_k[i];
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, m, &minusOne, F, 
        N, Gamma, 1, &one, f_wavg, 1);

    free(Gamma);
    
    // add beta * f to x_{k+1}
    for (i = 0; i < N; i++)
        x_kp1[i] += betaA * f_wavg[i];
    
    free(f_wavg);
}