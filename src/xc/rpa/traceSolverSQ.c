/**
 * @file    traceSolverSQ.c
 * @brief   This file saves functions used for SQ_trace_estimator and SQ_trace_estimator_kpt
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <complex.h>

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

#include "tools.h"
#include "tools_RPA.h"

void transfer_Hp_HpDiagComms(int nuChi0Neig, double *Hp, int *input_desc_Hp_BLCYC, double *HpSQ, int *input_desc_HpSQ, int nrHpSQ, int ncHpSQ,
    int nuChi0EigsBridgeCommIndex, int HpDiagCommIndex, int ctxtWorld, MPI_Comm HpDiagBridgeComm, int rankHpDiagComm, int printFlag) {
    int desc_Hp_BLCYC[9], desc_HpSQ[9];
    if (!nuChi0EigsBridgeCommIndex) {
        for (int i = 0; i < 9; i++) {
            desc_Hp_BLCYC[i] = input_desc_Hp_BLCYC[i];
        }
    } else {
        desc_Hp_BLCYC[1] = -1;
    }
    if (!HpDiagCommIndex) {
        for (int i = 0; i < 9; i++) {
            desc_HpSQ[i] = input_desc_HpSQ[i];
        }
    } else {
        desc_HpSQ[1] = -1;
    }
    int ONE = 1;
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
    int globalrank; double t1, t2, d2dtime, bcasttime;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);
    t1 = MPI_Wtime();
    #endif
    pdgemr2d_(&nuChi0Neig, &nuChi0Neig, Hp, &ONE, &ONE, desc_Hp_BLCYC,
        HpSQ, &ONE, &ONE, desc_HpSQ, &ctxtWorld);
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
    t2 = MPI_Wtime();
    d2dtime = t2 - t1;
    t1 = MPI_Wtime();
    #endif
    if (HpDiagCommIndex != -1) {
        MPI_Bcast(HpSQ, nrHpSQ * ncHpSQ, MPI_DOUBLE, 0, HpDiagBridgeComm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
    t2 = MPI_Wtime();
    bcasttime = t2 - t1;
    if (!globalrank) printf("in function transfer_Hp_HpDiagComms, pdgemr2d_ spent %.3F ms, MPI_Bcast spent %.3F ms\n", d2dtime*1e3, bcasttime*1e3);
    #endif
    if ((rankHpDiagComm != -1) && printFlag && (nrHpSQ * ncHpSQ > 1)) {
        char HpSQname[50];
        snprintf(HpSQname, 50, "rank%dHpDiagComm%d_Hp.txt", rankHpDiagComm, HpDiagCommIndex);
        FILE *HpSQfile = fopen(HpSQname, "a");
        for (int row = 0; row < nrHpSQ; row++) {
            for (int col = 0; col < ncHpSQ; col++) {
                fprintf(HpSQfile, "%12.9f ", HpSQ[col*nrHpSQ + row]);
            }
            fprintf(HpSQfile, "\n");
        }
        fprintf(HpSQfile, "\n");
        fclose(HpSQfile);
    }
}

int Lanczos_decompose_Hp_group(int nuChi0Neig, double *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex, 
    int vkStart, int vkEnd, int nrvk, int *desc_vk, int nrvk_BLCYC, int *desc_vk_BLCYC, int *pictxt_HpSQ,
    int nplLanczos, double *TdiagArrayGroup, double *ToffDiagArrayGroup, MPI_Comm HpDiagComm, int HpDiagCommIndex, int printFlag) {
    int nprocHpDiagComm, rankHpDiagComm;
    MPI_Comm_size(HpDiagComm, &nprocHpDiagComm);
    MPI_Comm_rank(HpDiagComm, &rankHpDiagComm);
    int finalNpl = 0, ONE = 1, nHpDiagComm = HpDiagEndIndex - HpDiagStartIndex + 1;
    double *vk = (double*)calloc(sizeof(double), nrvk*nHpDiagComm);
    double *vk_BLCYC;
    for (int diagIndex = HpDiagStartIndex; diagIndex < HpDiagEndIndex + 1; diagIndex++) {
        if ((diagIndex >= vkStart) && (diagIndex <= vkEnd)) {
            vk[nrvk*(diagIndex - HpDiagStartIndex) + diagIndex - vkStart] = 1.0;
        }
    }
    if (nprocHpDiagComm > 1) {
        vk_BLCYC = (double*)calloc(sizeof(double), nrvk_BLCYC*nHpDiagComm);
        pdgemr2d_(&nuChi0Neig, &nHpDiagComm, vk, &ONE, &ONE, desc_vk,
            vk_BLCYC, &ONE, &ONE, desc_vk_BLCYC, pictxt_HpSQ); 
    } else {
        vk_BLCYC = vk;
    }
    double *vkm1_BLCYC = (double*)calloc(sizeof(double), nrvk_BLCYC*nHpDiagComm);
    double *w_BLCYC = (double*)calloc(sizeof(double), nrvk_BLCYC*nHpDiagComm);
    int desc_w_BLCYC[9];
    for (int i = 0; i < 9; i++) desc_w_BLCYC[i] = desc_vk_BLCYC[i];
    double *b = (double*)calloc(sizeof(double), nHpDiagComm);
    double bnorm;
    double *localDotProduct = (double*)calloc(sizeof(double), nHpDiagComm);

    double alpha = 1.0, beta = 0.0;
    for (int count = 0; count < nplLanczos; count++) {
        finalNpl = count + 1;
        if (nprocHpDiagComm > 1) {
            pdgemm_("T", "N", &nuChi0Neig, &nHpDiagComm, &nuChi0Neig, &alpha, 
                HpSQ, &ONE, &ONE, desc_HpSQ, 
                vk_BLCYC, &ONE, &ONE, desc_vk_BLCYC, 
                &beta, w_BLCYC, &ONE, &ONE, desc_w_BLCYC);
        } else {
            cblas_dgemm(
                CblasColMajor, CblasTrans, CblasNoTrans,
                nuChi0Neig, nHpDiagComm, nuChi0Neig,
                1.0, HpSQ, nuChi0Neig, vk_BLCYC, nuChi0Neig, 
                0.0, w_BLCYC, nuChi0Neig
            );
        }
        for (int col = 0; col < nHpDiagComm; col++) {
            for (int row = 0; row < nrvk_BLCYC; row++) {
                w_BLCYC[col*nrvk_BLCYC + row] -= b[col]*vkm1_BLCYC[col*nrvk_BLCYC + row];
            }
            localDotProduct[col] = cblas_ddot(nrvk_BLCYC, vk_BLCYC + col*nrvk_BLCYC, ONE, w_BLCYC + col*nrvk_BLCYC, ONE); // no need to use pddot! We can make it if the intervals of the two vectors distributed in the processors correspond to each other
        }

        MPI_Allreduce(&localDotProduct[0], &TdiagArrayGroup[count*nHpDiagComm], nHpDiagComm, MPI_DOUBLE, MPI_SUM, HpDiagComm); // column major, col n: nth diagonal entries of all vectors row m: all diagonal entries of vector m
        if (count != nplLanczos - 1) {
            bnorm = 0.0;
            for (int col = 0; col < nHpDiagComm; col++) {
                for (int row = 0; row < nrvk_BLCYC; row++) {
                    w_BLCYC[col*nrvk_BLCYC + row] -= TdiagArrayGroup[count*nHpDiagComm + col] * vk_BLCYC[col*nrvk_BLCYC + row];
                }
                Vector2Norm(w_BLCYC + col*nrvk_BLCYC, nrvk_BLCYC, &b[col], HpDiagComm);
                bnorm += b[col]*b[col];
            }
            bnorm = sqrt(bnorm);
            if (bnorm < nHpDiagComm*1e-11) {
                break;
            }
            memcpy(ToffDiagArrayGroup + count*nHpDiagComm, b, sizeof(double)*nHpDiagComm);
            for (int col = 0; col < nHpDiagComm; col++) {
                for (int row = 0; row < nrvk_BLCYC; row++) {
                    vkm1_BLCYC[col*nrvk_BLCYC + row] = vk_BLCYC[col*nrvk_BLCYC + row];
                    vk_BLCYC[col*nrvk_BLCYC + row] = w_BLCYC[col*nrvk_BLCYC + row] / b[col];
                }
            }
        }
    }

    if (printFlag) {
        char HpSQname[50];
        snprintf(HpSQname, 50, "rank%dHpDiagComm%d_Hp.txt", rankHpDiagComm, HpDiagCommIndex);
        FILE *HpSQfile = fopen(HpSQname, "a");
        fprintf(HpSQfile, "diagIndex %d~%d, TdiagArray:\n", HpDiagStartIndex, HpDiagEndIndex);
        for (int col = 0; col < finalNpl; col++) {
            for (int row = 0; row < nHpDiagComm; row++) {
                fprintf(HpSQfile, "%12.9f ", TdiagArrayGroup[col*nHpDiagComm + row]);
            }
            fprintf(HpSQfile, "\n");
        }
        
        fprintf(HpSQfile, "\n");
        fprintf(HpSQfile, "ToffDiagArray:\n");
        for (int col = 0; col < finalNpl - 1; col++) {
            for (int row = 0; row < nHpDiagComm; row++) {
                fprintf(HpSQfile, "%12.9f ", ToffDiagArrayGroup[col*nHpDiagComm + row]);
            }
            fprintf(HpSQfile, "\n");
        }
        fprintf(HpSQfile, "\n");
        fclose(HpSQfile);
    }

    free(vk);
    if (nprocHpDiagComm > 1) {
        free(vk_BLCYC);
    }
    free(vkm1_BLCYC);
    free(w_BLCYC);
    free(b);
    free(localDotProduct);
    return finalNpl;
}

double diagonalize_T_get_diag_fHp(int finalNpl, int diagIndex, double *TdiagArray, double *ToffDiagArray, double *TeigVals, double *TeigVecs, 
    int rankHpDiagComm, int HpDiagCommIndex, int printFlag) { // there is no parallelization
    LAPACKE_dstev(LAPACK_COL_MAJOR,'V',finalNpl,TdiagArray,ToffDiagArray,TeigVecs,finalNpl);    
    double diagResult = 0.0;
    for (int i = 0; i < finalNpl; i++) {
        diagResult += TeigVecs[i*finalNpl] * (log(1.0 - TdiagArray[i]) + TdiagArray[i]) * TeigVecs[i*finalNpl];
    }
    if (printFlag) {
        char HpSQname[50];
        snprintf(HpSQname, 50, "rank%dHpDiagComm%d_Hp.txt", rankHpDiagComm, HpDiagCommIndex);
        FILE *HpSQfile = fopen(HpSQname, "a");
        fprintf(HpSQfile, "diagIndex = %d, eigVecs of T:\n", diagIndex);
        for (int row = 0; row < finalNpl; row++) {
            for (int col = 0; col < finalNpl; col++) {
                fprintf(HpSQfile, "%12.9f ", TeigVecs[col*finalNpl + row]);
            }
            fprintf(HpSQfile, "\n");
        }
        fprintf(HpSQfile, "\n");
        fprintf(HpSQfile, "diagIndex = %d, eigVals of T:\n", diagIndex);
        for (int count = 0; count < finalNpl; count++) {
            fprintf(HpSQfile, "%12.9f ", TeigVals[count]);
        }
        fprintf(HpSQfile, "\n");
        fprintf(HpSQfile, "fHp[%d] = %12.9f\n", diagIndex, diagResult);
        fclose(HpSQfile);
    }

    return diagResult;
}

void transfer_Hp_HpDiagComms_kpt(int nuChi0Neig, double _Complex *Hp, int *input_desc_Hp_BLCYC, double _Complex *HpSQ, int *input_desc_HpSQ, int nrHpSQ, int ncHpSQ,
    int nuChi0EigsBridgeCommIndex, int HpDiagCommIndex, int ctxtWorld, MPI_Comm HpDiagBridgeComm, int rankHpDiagComm, int printFlag) {
    int desc_Hp_BLCYC[9], desc_HpSQ[9];
    if (!nuChi0EigsBridgeCommIndex) {
        for (int i = 0; i < 9; i++) {
            desc_Hp_BLCYC[i] = input_desc_Hp_BLCYC[i];
        }
    } else {
        desc_Hp_BLCYC[1] = -1;
    }
    if (!HpDiagCommIndex) {
        for (int i = 0; i < 9; i++) {
            desc_HpSQ[i] = input_desc_HpSQ[i];
        }
    } else {
        desc_HpSQ[1] = -1;
    }
    int ONE = 1;
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
    int globalrank; double t1, t2, d2dtime, bcasttime;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);
    t1 = MPI_Wtime();
    #endif
    pzgemr2d_(&nuChi0Neig, &nuChi0Neig, Hp, &ONE, &ONE, desc_Hp_BLCYC,
        HpSQ, &ONE, &ONE, desc_HpSQ, &ctxtWorld);
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
    t2 = MPI_Wtime();
    d2dtime = t2 - t1;
    t1 = MPI_Wtime();
    #endif
    if (HpDiagCommIndex != -1) {
        MPI_Bcast(HpSQ, nrHpSQ * ncHpSQ, MPI_DOUBLE_COMPLEX, 0, HpDiagBridgeComm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
    t2 = MPI_Wtime();
    bcasttime = t2 - t1;
    if (!globalrank) printf("in function transfer_Hp_HpDiagComms, pdgemr2d_ spent %.3F ms, MPI_Bcast spent %.3F ms\n", d2dtime*1e3, bcasttime*1e3);
    #endif
    if ((rankHpDiagComm != -1) && printFlag && (nrHpSQ * ncHpSQ > 1)) {
        char HpSQname[50];
        snprintf(HpSQname, 50, "rank%dHpDiagComm%d_Hp_kpt.txt", rankHpDiagComm, HpDiagCommIndex);
        FILE *HpSQfile = fopen(HpSQname, "a");
        for (int row = 0; row < nrHpSQ; row++) {
            for (int col = 0; col < ncHpSQ; col++) {
                fprintf(HpSQfile, "%12.9f + %12.9fi ", creal(HpSQ[col*nrHpSQ + row]), cimag(HpSQ[col*nrHpSQ + row]));
            }
            fprintf(HpSQfile, "\n");
        }
        fprintf(HpSQfile, "\n");
        fclose(HpSQfile);
    }
}

int Lanczos_decompose_Hp_group_kpt(int nuChi0Neig, double _Complex *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex, 
    int vkStart, int vkEnd, int nrvk, int *desc_vk, int nrvk_BLCYC, int *desc_vk_BLCYC, int *pictxt_HpSQ,
    int nplLanczos, double _Complex *TdiagArrayGroup, double _Complex *ToffDiagArrayGroup, MPI_Comm HpDiagComm, int HpDiagCommIndex, int printFlag) {
    int nprocHpDiagComm, rankHpDiagComm;
    MPI_Comm_size(HpDiagComm, &nprocHpDiagComm);
    MPI_Comm_rank(HpDiagComm, &rankHpDiagComm);
    int finalNpl = 0, ONE = 1, nHpDiagComm = HpDiagEndIndex - HpDiagStartIndex + 1;
    double _Complex *vk = (double _Complex*)calloc(sizeof(double _Complex), nrvk*nHpDiagComm);
    double _Complex *vk_BLCYC;
    for (int diagIndex = HpDiagStartIndex; diagIndex < HpDiagEndIndex + 1; diagIndex++) {
        if ((diagIndex >= vkStart) && (diagIndex <= vkEnd)) {
            vk[nrvk*(diagIndex - HpDiagStartIndex) + diagIndex - vkStart] = 1.0;
        }
    }
    if (nprocHpDiagComm > 1) {
        vk_BLCYC = (double _Complex*)calloc(sizeof(double _Complex), nrvk_BLCYC*nHpDiagComm);
        pzgemr2d_(&nuChi0Neig, &nHpDiagComm, vk, &ONE, &ONE, desc_vk,
            vk_BLCYC, &ONE, &ONE, desc_vk_BLCYC, pictxt_HpSQ); 
    } else {
        vk_BLCYC = vk;
    }
    double _Complex *vkm1_BLCYC = (double _Complex*)calloc(sizeof(double _Complex), nrvk_BLCYC*nHpDiagComm);
    double _Complex *w_BLCYC = (double _Complex*)calloc(sizeof(double _Complex), nrvk_BLCYC*nHpDiagComm);
    int desc_w_BLCYC[9];
    for (int i = 0; i < 9; i++) desc_w_BLCYC[i] = desc_vk_BLCYC[i];
    double *b = (double*)calloc(sizeof(double), nHpDiagComm);
    double bnorm;
    double _Complex *localDotProduct = (double _Complex*)calloc(sizeof(double _Complex), nHpDiagComm);

    double _Complex alpha = 1.0, beta = 0.0;
    for (int count = 0; count < nplLanczos; count++) {
        finalNpl = count + 1;
        if (nprocHpDiagComm > 1) {
            pzgemm_("T", "N", &nuChi0Neig, &nHpDiagComm, &nuChi0Neig, &alpha, 
                HpSQ, &ONE, &ONE, desc_HpSQ, 
                vk_BLCYC, &ONE, &ONE, desc_vk_BLCYC, 
                &beta, w_BLCYC, &ONE, &ONE, desc_w_BLCYC);
        } else {
            cblas_zgemm(
                CblasColMajor, CblasTrans, CblasNoTrans,
                nuChi0Neig, nHpDiagComm, nuChi0Neig,
                &alpha, HpSQ, nuChi0Neig, vk_BLCYC, nuChi0Neig, 
                &beta, w_BLCYC, nuChi0Neig
            );
        }
        for (int col = 0; col < nHpDiagComm; col++) {
            for (int row = 0; row < nrvk_BLCYC; row++) {
                w_BLCYC[col*nrvk_BLCYC + row] -= b[col]*vkm1_BLCYC[col*nrvk_BLCYC + row];
            }
            cblas_zdotc_sub(nrvk_BLCYC, vk_BLCYC + col*nrvk_BLCYC, ONE, w_BLCYC + col*nrvk_BLCYC, ONE, &(localDotProduct[col])); // no need to use pddot! We can make it if the intervals of the two vectors distributed in the processors correspond to each other
        }

        MPI_Allreduce(&localDotProduct[0], &TdiagArrayGroup[count*nHpDiagComm], nHpDiagComm, MPI_DOUBLE_COMPLEX, MPI_SUM, HpDiagComm); // column major, col n: nth diagonal entries of all vectors row m: all diagonal entries of vector m
        if (count != nplLanczos - 1) {
            bnorm = 0.0;
            for (int col = 0; col < nHpDiagComm; col++) {
                for (int row = 0; row < nrvk_BLCYC; row++) {
                    w_BLCYC[col*nrvk_BLCYC + row] -= TdiagArrayGroup[count*nHpDiagComm + col] * vk_BLCYC[col*nrvk_BLCYC + row];
                }
                Vector2Norm_complex(w_BLCYC + col*nrvk_BLCYC, nrvk_BLCYC, &b[col], HpDiagComm);
                bnorm += b[col]*b[col];
            }
            bnorm = sqrt(bnorm);
            if (bnorm < nHpDiagComm*1e-11) {
                break;
            }
            for (int col = 0; col < nHpDiagComm; col++) {
                ToffDiagArrayGroup[count*nHpDiagComm + col] = b[col];
                for (int row = 0; row < nrvk_BLCYC; row++) {
                    vkm1_BLCYC[col*nrvk_BLCYC + row] = vk_BLCYC[col*nrvk_BLCYC + row];
                    vk_BLCYC[col*nrvk_BLCYC + row] = w_BLCYC[col*nrvk_BLCYC + row] / b[col];
                }
            }
        }
    }

    if (printFlag) {
        char HpSQname[50];
        snprintf(HpSQname, 50, "rank%dHpDiagComm%d_Hp_kpt.txt", rankHpDiagComm, HpDiagCommIndex);
        FILE *HpSQfile = fopen(HpSQname, "a");
        fprintf(HpSQfile, "diagIndex %d~%d, TdiagArray:\n", HpDiagStartIndex, HpDiagEndIndex);
        for (int col = 0; col < finalNpl; col++) {
            for (int row = 0; row < nHpDiagComm; row++) {
                fprintf(HpSQfile, "%12.9f + %12.9fi ", creal(TdiagArrayGroup[col*nHpDiagComm + row]), cimag(TdiagArrayGroup[col*nHpDiagComm + row]));
            }
            fprintf(HpSQfile, "\n");
        }
        
        fprintf(HpSQfile, "\n");
        fprintf(HpSQfile, "ToffDiagArray:\n");
        for (int col = 0; col < finalNpl - 1; col++) {
            for (int row = 0; row < nHpDiagComm; row++) {
                fprintf(HpSQfile, "%12.9f + %12.9fi ", creal(ToffDiagArrayGroup[col*nHpDiagComm + row]), cimag(ToffDiagArrayGroup[col*nHpDiagComm + row]));
            }
            fprintf(HpSQfile, "\n");
        }
        fprintf(HpSQfile, "\n");
        fclose(HpSQfile);
    }

    free(vk);
    if (nprocHpDiagComm > 1) {
        free(vk_BLCYC);
    }
    free(vkm1_BLCYC);
    free(w_BLCYC);
    free(b);
    free(localDotProduct);
    return finalNpl;
}

double diagonalize_T_get_diag_fHp_kpt(int finalNpl, int diagIndex, double _Complex *TdiagArray, double _Complex *ToffDiagArray, double *TeigVals, double _Complex *TeigVecs, 
    int rankHpDiagComm, int HpDiagCommIndex, int printFlag) { // there is no parallelization
    double _Complex *T = TeigVecs;
    for (int count = 0; count < finalNpl * finalNpl; count++) { // reset the value inside the array
        T[count] = 0.0;
    }
    for (int count = 0; count < finalNpl - 1; count++) {
        T[count*finalNpl + count] = TdiagArray[count];
        T[count*finalNpl + count + 1] = ToffDiagArray[count]; // Lower triangular
    }
    T[finalNpl*finalNpl - 1] = TdiagArray[finalNpl - 1];
    LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'L', finalNpl, T, finalNpl, TeigVals);

    double diagResult = 0.0;
    for (int i = 0; i < finalNpl; i++) {
        diagResult += TeigVecs[i*finalNpl] * (log(1.0 - TeigVals[i]) + TeigVals[i]) * conj(TeigVecs[i*finalNpl]);
    }

    if (printFlag) {
        char HpSQname[50];
        snprintf(HpSQname, 50, "rank%dHpDiagComm%d_Hp_kpt.txt", rankHpDiagComm, HpDiagCommIndex);
        FILE *HpSQfile = fopen(HpSQname, "a");
        fprintf(HpSQfile, "diagIndex = %d, eigVecs of T:\n", diagIndex);
        for (int row = 0; row < finalNpl; row++) {
            for (int col = 0; col < finalNpl; col++) {
                fprintf(HpSQfile, "%12.9f + %12.9fi ", creal(TeigVecs[col*finalNpl + row]), cimag(TeigVecs[col*finalNpl + row]));
            }
            fprintf(HpSQfile, "\n");
        }
        fprintf(HpSQfile, "\n");
        fprintf(HpSQfile, "diagIndex = %d, eigVals of T:\n", diagIndex);
        for (int count = 0; count < finalNpl; count++) {
            fprintf(HpSQfile, "%12.9f ", TeigVals[count]);
        }
        fprintf(HpSQfile, "\n");
        fprintf(HpSQfile, "fHp[%d] = %12.9f\n", diagIndex, diagResult);
        fclose(HpSQfile);
    }

    return diagResult;
}