/**
 * @file    eigenSolverKpt_RPA.c
 * @brief   This file saves functions used for subspace iteration, called by project_tildeChi_general_eigProblem_kpt and 
 *          project_tildeChi_update_deltaV_kpt
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
extern void pzherk_();
extern void pzpotrf_();
#endif

#include "eigenSolver.h"
#include "tools.h"
#include "linearAlgebra.h"

#include "restoreElectronicGroundState.h"
#include "subspaceIter.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverKpt_RPA.h"
#include "tools_RPA.h"


void YT_multiply_Y_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm, double _Complex *Y, int DMnd, int Nspinor_eig, int flagNoDmcomm, double _Complex *Y_BLCYC, double _Complex *Mp, int printFlag) {
    if (flagNoDmcomm) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rankWorld; // nproc_dmcomm_phi,
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int rank;
    MPI_Comm_rank(pRPA->nuChi0BlacsComm, &rank);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double t1, t2, t3, t4;
    #ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
    #endif
    int DMndspe = DMnd * Nspinor_eig;
    
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        int ONE = 1;
        double alpha = 1.0, beta = 0.0;

        t3 = MPI_Wtime();
        pzgemr2d_(&DMndspe, &pRPA->nuChi0Neig, Y, &ONE, &ONE, pRPA->desc_orbitals,
            Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &pRPA->ictxt_blacs); 
        t4 = MPI_Wtime();
        #ifdef DEBUG  
        if(!rankWorld) 
            printf("rank = %2d, Distribute orbital to block cyclic format took %.3f ms, STARTING PZSYRK ...\n", rankWorld, (t4 - t3)*1e3);          
        #endif
        // perform matrix multiplication using ScaLAPACK routines
        pzherk_("U", "C", &pRPA->nuChi0Neig, &DMndspe, &alpha, Y_BLCYC, &ONE, &ONE,
            pRPA->desc_orb_BLCYC, &beta, Mp, &ONE, &ONE, pRPA->desc_Mp_BLCYC);
    } else {
        #ifdef DEBUG    
        if (!rankWorld) printf("rank = %d, STARTING ZSYRK ...\n",rankWorld);
        #endif 
        cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, pRPA->nuChi0Neig, DMndspe, 1.0, 
            Y, DMndspe, 0.0, Mp, pRPA->nuChi0Neig);
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rankWorld) 
        printf("rank = %2d, YT'*Y in block cyclic format in each blacscomm took %.3f ms\n", 
                rankWorld, (t2 - t1)*1e3); 
    #endif
    
    if (printFlag && (pRPA->nr_Hp_BLCYC > 1)) {
        char Mpname[50];
        snprintf(Mpname, 50, "rank%dblacsComm%d_Mp.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
        FILE *Mpfile = fopen(Mpname, "a");
        for (int row = 0; row < pRPA->nr_Mp_BLCYC; row++) {
            for (int col = 0; col < pRPA->nc_Mp_BLCYC; col++) {
                fprintf(Mpfile, "%12.9f+%12.9fi", creal(Mp[col*pRPA->nr_Mp_BLCYC + row]), cimag(Mp[col*pRPA->nr_Mp_BLCYC + row]));
            }
            fprintf(Mpfile, "\n");
        }
        fprintf(Mpfile, "\n");
        fclose(Mpfile);
    }
    #ifdef DEBUG
    et = MPI_Wtime();
    if (!rankWorld) printf("Rank 0, YT*Y used %.3lf ms\n", 1000.0 * (et - st)); 
    #endif
}

void Y_orth_permute_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, double _Complex *Ys_BLCYC, double _Complex *Mp, double _Complex *Ys_orthed, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
    double t1, t2, t3;
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int ONE = 1, info;
    int NYrows = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_eig;
    int NYcols = pRPA->nuChi0Neig;
    // Orthogonalization using Choleskey 
    t1 = MPI_Wtime();

    // Cholesky factorization
    double _Complex alpha = 1.0, beta = 0.0;
    pzpotrf_("U", &NYcols, Mp, &ONE, &ONE, pRPA->desc_Mp_BLCYC, &info);  
    pztrsm_("R", "U", "N", "N", &NYrows, &NYcols, &alpha, Mp, &ONE, &ONE, pRPA->desc_Mp_BLCYC, Ys_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC); // PROBLEM!

    double _Complex *Ys_BLCYC_permuted;
    if (pRPA->npnuChi0Neig > 1 && pRPA->flagCyclicPermute) { // permute vectors
        Ys_BLCYC_permuted = (double _Complex*)malloc(sizeof(double _Complex) * pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC);
        
        pzgemm_("N", "N", &NYrows, &NYcols, &NYcols, &alpha, 
            Ys_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
            pRPA->permute_BLCYC_kpt, &ONE, &ONE, pRPA->desc_permute_BLCYC, 
            &beta, Ys_BLCYC_permuted, &ONE, &ONE, pRPA->desc_orb_BLCYC);
        memcpy(Ys_BLCYC, Ys_BLCYC_permuted, sizeof(double _Complex) * pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC);
        free(Ys_BLCYC_permuted);
    }
    
    t2 = MPI_Wtime();
    // update Ys
    pzgemr2d_(&NYrows, &NYcols, Ys_BLCYC, &ONE, &ONE, 
        pRPA->desc_orb_BLCYC, Ys_orthed, &ONE, &ONE, 
        pRPA->desc_orbitals, &pRPA->ictxt_blacs);
    t3 = MPI_Wtime();

    double _Complex modulefirstYOrthed = 0.0;
    for (int index = 0; index < NYrows; index++) {
        modulefirstYOrthed += Ys_orthed[index]*Ys_orthed[index];
    }
    if ((cabs(modulefirstYOrthed) > 2.0) && (!rankWorld)) {
        printf("ERROR: vector after orthogonalization diverged!\n");
    }
    #ifdef DEBUG
    if(!rankWorld) printf("global rank %d, Orthogonalization of orbitals took: %.3f ms\n", rankWorld, (t2 - t1)*1e3); 
    if(!rankWorld) printf("global rank %d, Updating orbitals took: %.3f ms\n", rankWorld, (t3 - t2)*1e3);
    #endif
}

void project_YT_nuChi0_Y_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm, double _Complex *Y_BLCYC, double _Complex *HY, int DMnd, int Nspinor_eig, int flagNoDmcomm, double _Complex *Hp, double _Complex *outputHY_BLCYC, int printFlag) {
    if (flagNoDmcomm) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rankWorld; // nproc_dmcomm_phi,
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int rank;
    MPI_Comm_rank(pRPA->nuChi0BlacsComm, &rank);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double t1, t2;
    #ifdef DEBUG
    double st, et;   
    st = MPI_Wtime();
    #endif
    int DMndspe = DMnd * Nspinor_eig;
    int ONE = 1;

    double _Complex alpha = 1.0, beta = 0.0;

    double _Complex *HY_BLCYC;
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        // distribute HY
        HY_BLCYC = (double _Complex *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
        assert(HY_BLCYC != NULL);
        pzgemr2d_(&DMndspe, &pRPA->nuChi0Neig, HY, &ONE, &ONE, pRPA->desc_orbitals, 
            HY_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &pRPA->ictxt_blacs);
    } else {
        HY_BLCYC = HY;
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, distributing HY into block cyclic form took %.3f ms\n", 
                 rankWorld, (t2 - t1)*1e3);  
    #endif
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        // perform matrix multiplication Y' * HY using ScaLAPACK routines
        pzgemm_("C", "N", &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &DMndspe, &alpha, 
            Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
            HY_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
            &beta, Hp, &ONE, &ONE, pRPA->desc_Hp_BLCYC);
    } else {
        cblas_zgemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            pRPA->nuChi0Neig, pRPA->nuChi0Neig, DMndspe,
            &alpha, Y_BLCYC, DMndspe, HY_BLCYC, DMndspe, 
            &beta, Hp, pRPA->nuChi0Neig
        );
    }

    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, finding Y'*HY took %.3f ms\n",rankWorld,(t2-t1)*1e3); 
    #endif
    if (size_blacscomm > 1) {
        memcpy(outputHY_BLCYC, HY_BLCYC, pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
        free(HY_BLCYC);
    }

    if (printFlag && (pRPA->nr_Hp_BLCYC > 1)) {
        char Hpname[50];
        snprintf(Hpname, 50, "rank%dblacsComm%d_Hp_kpt.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
        FILE *Hpfile = fopen(Hpname, "a");
        for (int row = 0; row < pRPA->nr_Hp_BLCYC; row++) {
            for (int col = 0; col < pRPA->nc_Hp_BLCYC; col++) {
                fprintf(Hpfile, "%12.9f + %12.9fi ", creal(Hp[col*pRPA->nr_Hp_BLCYC + row]), cimag(Hp[col*pRPA->nr_Hp_BLCYC + row]));
            }
            fprintf(Hpfile, "\n");
        }
        fprintf(Hpfile, "\n");
        fclose(Hpfile);
    }
    #ifdef DEBUG
    et = MPI_Wtime();
    if (!rankWorld) printf("Rank 0, project_nuChi0 used %.3lf ms\n", 1000.0 * (et - st)); 
    #endif
// // #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}

void generalized_eigenproblem_solver_kpt(RPA_OBJ* pRPA, MPI_Comm blacsComm, 
    double _Complex *Hp, double _Complex *Mp, int nr, int nc, int *descHp, int *descMp,
    double *eigValues, double _Complex *Q, int nrQ, int ncQ, int *descQ,
    int blksz, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rankWorld; // nproc_dmcomm_phi, 
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    double t1, t2;
    #ifdef DEBUG    
    double st = MPI_Wtime();
    #endif
    if (pRPA->eig_useLAPACK == 1) { // solve generalized eigenproblem Hp X = Mp X \Lambda, where Hp and Mp are symmetric
        int info = 0;
        t1 = MPI_Wtime();
        if ((nr == pRPA->nuChi0Neig) && (nc == pRPA->nuChi0Neig)) {
            info = LAPACKE_zhegvd(LAPACK_COL_MAJOR, 1, 'V', 'U', pRPA->nuChi0Neig, 
                Hp, pRPA->nuChi0Neig, Mp, pRPA->nuChi0Neig, eigValues);
            printf("global rank %d, the first 3 eigenvalues are %.5E, %.5E, %.5E\n", rankWorld, eigValues[0], eigValues[1], eigValues[2]);
            int nuChi0EigsCommRank;
            MPI_Comm_rank(pRPA->nuChi0Eigscomm, &nuChi0EigsCommRank);
        }
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rankWorld) {
            printf("==standard eigenproblem: "
                "info = %d, solving standard eigenproblem using LAPACKE_dggev: %.3f ms\n", 
                info, (t2 - t1)*1e3);
        }
        #endif
        int ONE = 1;
        t1 = MPI_Wtime();
        // distribute eigenvectors to block cyclic format
        pzgemr2d_(&pRPA->nuChi0Neig, &pRPA->nuChi0Neig, Hp, &ONE, &ONE, 
                descHp, Q, &ONE, &ONE, 
                descQ, &pRPA->ictxt_blacs_topo);
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rankWorld) {
            printf("==generalized eigenproblem: "
                "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
                (t2 - t1)*1e3);
        }
        #endif
    } else {
        int rank, nproc;
        MPI_Comm_rank(blacsComm, &rank);
        MPI_Comm_size(blacsComm, &nproc);

        int ONE = 1, il = 1, iu = 1, *ifail, info, N, M, NZ;
        double vl = 0.0, vu = 0.0, abstol, orfac;

        ifail = (int *)malloc(pRPA->nuChi0Neig * sizeof(int));
        N = pRPA->nuChi0Neig;
        orfac = 0.0;
        #ifdef DEBUG
        if(!rank) printf("rank = %d, orfac = %.3e\n", rank, orfac);
        #endif
        // this setting yields the most orthogonal eigenvectors
        // abstol = pdlamch_(&pSPARC->ictxt_blacs_topo, "U");
        abstol = -1.0;
        pzhegvx_subcomm_ (&ONE, "V", "A", "U", &N, 
                    Hp, &ONE, &ONE, descHp, 
                    Mp, &ONE, &ONE, descMp, 
                    &vl, &vu, &il, &iu, &abstol, 
                    &M, &NZ, eigValues, &orfac, 
                    Q, &ONE, &ONE, descQ, 
                    ifail, &info, blacsComm, pRPA->eig_paral_subdims, blksz);

        // if (!rank){
        //     FILE *eigfile = fopen("eig.txt", "a");
        //     for (int col = 0; col < pRPA->nuChi0Neig; col++) {
        //         fprintf(eigfile, "%12.9f ", eigValues[col]);
        //     }
        //     fprintf(eigfile, "\n");
        //     fclose(eigfile);
        // }
        if (printFlag) {
            if (nrQ > 1){
                char Qfile[50];
                snprintf(Qfile, 50, "rank%dblacsComm%d_eigVec_Q_kpt.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
                FILE *eigVecfile = fopen(Qfile, "a");
                fprintf(eigVecfile, "\n");
                for (int row = 0; row < nrQ; row++) {
                    for (int col = 0; col < ncQ; col++) {
                        fprintf(eigVecfile, "%12.9f + %12.9fi ", creal(pRPA->Q[col*nrQ + row]), cimag(pRPA->Q[col*nrQ + row]));
                    }
                    fprintf(eigVecfile, "\n");
                }
                fclose(eigVecfile);
            }
        }
    }
    #ifdef DEBUG    
    double et = MPI_Wtime();
    if (rankWorld == 0) printf("global rank = %d, generalized_eigenproblem_solver_gamma used %.3lf ms\n", rankWorld, 1000.0 * (et - st));
    #endif
}

void subspace_rotation_update_deltaVs_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm, int DMnd, int Nspinor_eig, 
    double _Complex *nuChi0Y_BLCYC, double _Complex *deltaVs, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
    int ONE = 1;
    int DMndspe = DMnd * Nspinor_eig;

    #ifdef DEBUG
    double st = MPI_Wtime();
    #endif
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double _Complex alpha = 1.0, beta = 0.0;
    double t1, t2;

    t1 = MPI_Wtime();
    // double *Y_BLCYC = pRPA->Ys_BLCYC;
    double _Complex *Q = pRPA->Q_kpt;
    double _Complex *X_BLCYC;
    if (size_blacscomm > 1) {
        // perform matrix multiplication Psi * Q using ScaLAPACK routines
        X_BLCYC = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
        pzgemm_("N", "N", &DMndspe, &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &alpha, 
                nuChi0Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                Q, &ONE, &ONE, pRPA->desc_Q_BLCYC, 
                &beta, X_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC);
        if (pRPA->flagCyclicPermute) {
            double _Complex *X_BLCYC_permuted = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
            pzgemm_("N", "N", &DMndspe, &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &alpha, 
                    X_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                    pRPA->permute_BLCYC_kpt, &ONE, &ONE, pRPA->desc_permute_BLCYC, 
                    &beta, X_BLCYC_permuted, &ONE, &ONE, pRPA->desc_orb_BLCYC);
            memcpy(X_BLCYC, X_BLCYC_permuted, pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
            free(X_BLCYC_permuted);
        }
    } else {
        cblas_zgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            DMndspe, pRPA->nuChi0Neig, pRPA->nuChi0Neig, 
            &alpha, nuChi0Y_BLCYC, DMndspe, Q, pRPA->nuChi0Neig,
            &beta, deltaVs, DMndspe
        );
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, subspace rotation took %.3f ms\n", rankWorld, (t2 - t1)*1e3); 
    #endif

    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        // distribute rotated orbitals from block cyclic format back into 
        // original format (band + domain)
        pzgemr2d_(&DMndspe, &pRPA->nuChi0Neig, X_BLCYC, &ONE, &ONE, 
                  pRPA->desc_orb_BLCYC, deltaVs, &ONE, &ONE, 
                  pRPA->desc_orbitals, &pRPA->ictxt_blacs);
        free(X_BLCYC);
    }
    t2 = MPI_Wtime();    
    #ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, Distributing orbital back into band + domain format took %.3f ms\n", rankWorld, (t2 - t1)*1e3);
    #endif

    #ifdef DEBUG
    double et = MPI_Wtime();
     if(!rankWorld) printf("global rank = %d, Subspace_Rotation used %.3lf ms\n\n", rankWorld, 1000.0 * (et - st));
    #endif
}
