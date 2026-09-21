/**
 * @file    eigenSolverGamma_RPA.c
 * @brief   This file saves functions used for subspace iteration called by project_tildeChi_general_eigProblem and
 *          project_tildeChi_update_deltaV
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
#endif

#include "eigenSolver.h"
#include "tools.h"
#include "linearAlgebra.h"

#include "restoreElectronicGroundState.h"
#include "subspaceIter.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "tools_RPA.h"

#define max(x,y) (((x) > (y)) ? (x) : (y))
#define min(x,y) (((x) > (y)) ? (y) : (x))


void YT_multiply_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm, double *Y, int DMnd, int Nspinor_eig, int flagNoDmcomm, double *Y_BLCYC, double *Mp, int printFlag) {
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
    // perform matrix multiplication using ScaLAPACK routines
    if (size_blacscomm > 1) { 
        int ONE = 1;
        double alpha = 1.0, beta = 0.0;

        t3 = MPI_Wtime();
        pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, Y, &ONE, &ONE, pRPA->desc_orbitals,
            Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &pRPA->ictxt_blacs); 
        t4 = MPI_Wtime();  
        #ifdef DEBUG  
        if(!rankWorld) 
            printf("global rank = %2d, Distribute Y to block cyclic format took %.3f ms, STARTING PDGEMM ...\n", rankWorld, (t4 - t3)*1e3);          
        #endif
        // perform matrix multiplication using ScaLAPACK routines
        pdsyrk_("U", "T", &pRPA->nuChi0Neig, &DMndspe, &alpha, Y_BLCYC, &ONE, &ONE,
            pRPA->desc_orb_BLCYC, &beta, Mp, &ONE, &ONE, pRPA->desc_Mp_BLCYC);
    } else {
        #ifdef DEBUG    
        if (!rankWorld) printf("global rank = %d, STARTING DGEMM...\n",rankWorld);
        #endif
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, pRPA->nuChi0Neig, DMndspe, 1.0, 
            Y, DMndspe, 0.0, Mp, pRPA->nuChi0Neig);
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rankWorld) 
        printf("global rank = %2d, YT'*Y in block cyclic format in each blacscomm took %.3f ms\n", 
                rankWorld, (t2 - t1)*1e3); 
    #endif

    if (printFlag && (pRPA->nr_Hp_BLCYC > 1)) {
        char Mpname[50];
        snprintf(Mpname, 50, "rank%dblacsComm%d_Mp.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
        FILE *Mpfile = fopen(Mpname, "a");
        for (int row = 0; row < pRPA->nr_Mp_BLCYC; row++) {
            for (int col = 0; col < pRPA->nc_Mp_BLCYC; col++) {
                fprintf(Mpfile, "%12.9f ", Mp[col*pRPA->nr_Mp_BLCYC + row]);
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

void Y_orth_permute_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, double *Ys_BLCYC, double *Mp, double *Ys_orthed, int flagNoDmcomm, int printFlag) {
    if (flagNoDmcomm) return;
    double t1, t2, t3;
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int ONE = 1;
    int NYrows = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_eig;
    int NYcols = pRPA->nuChi0Neig;
    // Orthogonalization using Choleskey 
    t1 = MPI_Wtime();
    Chol_orth(Ys_BLCYC, pRPA->desc_orb_BLCYC, Mp, pRPA->desc_Mp_BLCYC, &NYrows, &NYcols);

    double *Ys_BLCYC_permuted;
    if (pRPA->npnuChi0Neig > 1 && pRPA->flagCyclicPermute) { // permute vectors
        Ys_BLCYC_permuted = (double*)malloc(sizeof(double) * pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC);
        
        double alpha = 1.0, beta = 0.0; int ONE = 1;
        pdgemm_("N", "N", &NYrows, &NYcols, &NYcols, &alpha, 
            Ys_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
            pRPA->permute_BLCYC, &ONE, &ONE, pRPA->desc_permute_BLCYC, 
            &beta, Ys_BLCYC_permuted, &ONE, &ONE, pRPA->desc_orb_BLCYC);
        memcpy(Ys_BLCYC, Ys_BLCYC_permuted, sizeof(double) * pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC);
        free(Ys_BLCYC_permuted);
    }
    
    t2 = MPI_Wtime();
    // update Ys
    pdgemr2d_(&NYrows, &NYcols, Ys_BLCYC, &ONE, &ONE, 
        pRPA->desc_orb_BLCYC, Ys_orthed, &ONE, &ONE, 
        pRPA->desc_orbitals, &pRPA->ictxt_blacs);
    t3 = MPI_Wtime();
    if (printFlag) {
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                char afterOrthoName[100];
                snprintf(afterOrthoName, 100, "nuChi0Eigscomm%d_Ys_afterOrtho.txt", pRPA->nuChi0EigscommIndex);
                FILE *outputYs = fopen(afterOrthoName, "w");
                if (outputYs ==  NULL) {
                    printf("error printing deltaVs_afterOrtho\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int index = 0; index < NYrows; index++) {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < pRPA->nNuChi0Eigscomm; nuChi0EigIndex++) {
                            fprintf(outputYs, "%12.9f ", Ys_orthed[nuChi0EigIndex*NYrows + index]);
                        }
                        fprintf(outputYs, "\n");
                    }
                }
                fclose(outputYs);
            }
        }
    }
    
    double modulefirstYOrthed = 0.0;
    for (int index = 0; index < NYrows; index++) {
        modulefirstYOrthed += Ys_orthed[index]*Ys_orthed[index];
    }
    if ((modulefirstYOrthed > 2.0) && (!rankWorld)) {
        printf("ERROR: vector after orthogonalization diverged!\n");
    }
    #ifdef DEBUG
    if(!rankWorld) printf("global rank %d, Orthogonalization of orbitals took: %.3f ms\n", rankWorld, (t2 - t1)*1e3); 
    if(!rankWorld) printf("global rank %d, Updating orbitals took: %.3f ms\n", rankWorld, (t3 - t2)*1e3);
    #endif
}

void project_YT_nuChi0_Y_gamma(RPA_OBJ* pRPA, MPI_Comm dmcomm, double *Y_BLCYC, double *HY, int DMnd, int Nspinor_eig, int flagNoDmcomm, double *Hp, double *outputHY_BLCYC, int printFlag) {
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

    double alpha = 1.0, beta = 0.0;
    // double *HY = pRPA->deltaVs;
    // allocate memory for block cyclic format of the wavefunction
    // double *Y_BLCYC = pRPA->Ys_BLCYC;
    double *HY_BLCYC;
    t1 = MPI_Wtime();
    if (size_blacscomm > 1) {
        // distribute HY
        HY_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
        assert(HY_BLCYC != NULL);
        pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, HY, &ONE, &ONE, pRPA->desc_orbitals, 
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
        // FILE *outputFile;
        // if (printFlag) outputFile = fopen(pRPA->timeRecordname, "a");
        // fprintf(outputFile, "pRPA->nuChi0Neig %d, pRPA->nuChi0Neig %d, DMndspe %d, alpha %.3E\n", 
        //     pRPA->nuChi0Neig, pRPA->nuChi0Neig, DMndspe, alpha);
        // fprintf(outputFile, "Y_BLCYC[0] %.3E, pRPA->desc_orb_BLCYC %d %d %d %d %d %d %d %d %d\n", Y_BLCYC[0],
        //     pRPA->desc_orb_BLCYC[0], pRPA->desc_orb_BLCYC[1], pRPA->desc_orb_BLCYC[2], pRPA->desc_orb_BLCYC[3], pRPA->desc_orb_BLCYC[4], pRPA->desc_orb_BLCYC[5], pRPA->desc_orb_BLCYC[6], pRPA->desc_orb_BLCYC[7], pRPA->desc_orb_BLCYC[8]);
        // fprintf(outputFile, "HY_BLCYC[0] %.3E, pRPA->desc_orb_BLCYC %d %d %d %d %d %d %d %d %d\n", HY_BLCYC[0],
        //     pRPA->desc_orb_BLCYC[0], pRPA->desc_orb_BLCYC[1], pRPA->desc_orb_BLCYC[2], pRPA->desc_orb_BLCYC[3], pRPA->desc_orb_BLCYC[4], pRPA->desc_orb_BLCYC[5], pRPA->desc_orb_BLCYC[6], pRPA->desc_orb_BLCYC[7], pRPA->desc_orb_BLCYC[8]);
        // fprintf(outputFile, "beta %.3E, adress Hp %p, pRPA->desc_Hp_BLCYC %d %d %d %d %d %d %d %d %d\n", beta, Hp,
        //     pRPA->desc_Hp_BLCYC[0], pRPA->desc_Hp_BLCYC[1], pRPA->desc_Hp_BLCYC[2], pRPA->desc_Hp_BLCYC[3], pRPA->desc_Hp_BLCYC[4], pRPA->desc_Hp_BLCYC[5], pRPA->desc_Hp_BLCYC[6], pRPA->desc_Hp_BLCYC[7], pRPA->desc_Hp_BLCYC[8]);
        // fclose(outputFile);
        // perform matrix multiplication Y' * HY using ScaLAPACK routines
        pdgemm_("T", "N", &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &DMndspe, &alpha, 
                Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                HY_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, 
                &beta, Hp, &ONE, &ONE, pRPA->desc_Hp_BLCYC);
    } else {
        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans,
            pRPA->nuChi0Neig, pRPA->nuChi0Neig, DMndspe,
            1.0, Y_BLCYC, DMndspe, HY_BLCYC, DMndspe, 
            0.0, Hp, pRPA->nuChi0Neig
        );
    }

    t2 = MPI_Wtime();
    #ifdef DEBUG
    if(!rankWorld) printf("global rank = %2d, finding Y'*HY took %.3f ms\n",rankWorld,(t2-t1)*1e3); 
    #endif
    if (size_blacscomm > 1) {
        memcpy(outputHY_BLCYC, HY_BLCYC, pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
        free(HY_BLCYC);
    }

    if (printFlag && (pRPA->nr_Hp_BLCYC > 1)) {
        char Hpname[50];
        snprintf(Hpname, 50, "rank%dblacsComm%d_Hp.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
        FILE *Hpfile = fopen(Hpname, "a");
        for (int row = 0; row < pRPA->nr_Hp_BLCYC; row++) {
            for (int col = 0; col < pRPA->nc_Hp_BLCYC; col++) {
                fprintf(Hpfile, "%12.9f, ", Hp[col*pRPA->nr_Hp_BLCYC + row]);
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


void generalized_eigenproblem_solver_gamma(RPA_OBJ* pRPA, MPI_Comm blacsComm, 
    double *Hp, double *Mp, int nr, int nc, int *descHp, int *descMp,
    double *eigValues, double *Q, int nrQ, int ncQ, int *descQ, 
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
            info = LAPACKE_dsygvd(LAPACK_COL_MAJOR, 1, 'V', 'U', pRPA->nuChi0Neig, 
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
        pdgemr2d_(&pRPA->nuChi0Neig, &pRPA->nuChi0Neig, Hp, &ONE, &ONE, 
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
        pdsygvx_subcomm_ (&ONE, "V", "A", "U", &N, 
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
                snprintf(Qfile, 50, "rank%dblacsComm%d_eigVec_Q.txt", rank, pRPA->nuChi0EigsBridgeCommIndex);
                FILE *eigVecfile = fopen(Qfile, "a");
                fprintf(eigVecfile, "\n");
                for (int row = 0; row < nrQ; row++) {
                    for (int col = 0; col < ncQ; col++) {
                        fprintf(eigVecfile, "%12.9f ", pRPA->Q[col*nrQ + row]);
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
// #else // #if defined(USE_MKL) || defined(USE_SCALAPACK)

// #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}

// void automem_pzgeevx_( 
//     const char *balanc, const char *jobvl, const char *jobvr, const char *sense, 
//     const int *n, double _Complex *a, const int *desca, double _Complex *w, 
//     double _Complex *vl, const int *descvl, double _Complex *vr, const int *descvr, 
//     int *ilo, int *ihi, double *scale, double *abnrm, double *rconde, double *rcondv, int *info)
// {
// #if defined(USE_MKL) || defined(USE_SCALAPACK)
//     int grank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &grank);
// #ifdef DEBUG
//     double t1, t2;
// #endif

// 	int ictxt = desca[1], nprow, npcol, myrow, mycol;
// 	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

// 	int ZERO = 0, lwork;
// 	double _Complex *work;
// 	lwork = -1;
// 	work  = (double _Complex*)malloc(100 * sizeof(double _Complex));  

// 	//** first do a workspace query **//
// #ifdef DEBUG    
//     t1 = MPI_Wtime();
// #endif
// 	pzgeevx_(balanc, jobvl, jobvr, sense, n, a, desca, w, 
//      vl, descvl, vr, descvr, 
//      ilo, ihi, scale, abnrm, rconde, rcondv, 
//      work, &lwork, info);
// #ifdef DEBUG
//     t2 = MPI_Wtime();
//     if(!grank) printf("rank = %d, work(1) = %f, time for "
//                         "workspace query: %.3f ms\n", 
//                         grank, creal(work[0]), (t2 - t1)*1e3);
// #endif

// 	int NN, NP0, MQ0, NB, N = *n;
// 	lwork = (int) fabs(work[0]);
// 	NB = desca[4]; // distribution block size
// 	NN = max(max(N, NB),2);
// 	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
// 	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );

// 	lwork = max(lwork, 5 * N + max(5 * NN, NP0 * MQ0 + 2 * NB * NB) 
// 				+ ((N - 1) / (nprow * npcol) + 1) * NN);
// 	work = realloc(work, lwork * sizeof(double _Complex));

// 	// call the routine again to perform the calculation
// #ifdef DEBUG
//     t1 = MPI_Wtime();
// #endif
// 	pzgeevx_(balanc, jobvl, jobvr, sense, n, a, desca, w, 
//      vl, descvl, vr, descvr, 
//      ilo, ihi, scale, abnrm, rconde, rcondv, 
//      work, &lwork, info);
// #ifdef DEBUG
//     t2 = MPI_Wtime();
// if(!grank) {
//     printf("rank = %d, info = %d, time for solving standard "
//             "eigenproblem in %d x %d process grid: %.3f ms\n", 
//             grank, *info, nprow, npcol, (t2 - t1)*1e3);
//     printf("rank = %d, after calling pdgeevx, nuChi0Neig = %d\n", grank, *n);
// }
// #endif

// 	free(work);
// #endif // (USE_MKL or USE_SCALAPACK)	
// }

void subspace_rotation_update_deltaVs(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm, int DMnd, int Nspinor_eig, 
    double* nuChi0Y_BLCYC, double *deltaVs, int flagNoDmcomm, int printFlag) {
    int ONE = 1;
    int DMndspe = DMnd * Nspinor_eig;

    if (flagNoDmcomm) return;

    #ifdef DEBUG
    double st = MPI_Wtime();
    #endif
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    int size_blacscomm = pRPA->npnuChi0Neig;

    double alpha = 1.0, beta = 0.0;
    double t1, t2;

    t1 = MPI_Wtime();
    // double *Y_BLCYC = pRPA->Ys_BLCYC;
    double *Q = pRPA->Q;
    double *X_BLCYC;
    if (size_blacscomm > 1) {
        // perform matrix multiplication Psi * Q using ScaLAPACK routines
        X_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
        pdgemm_("N", "N", &DMndspe, &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &alpha, 
                nuChi0Y_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, Q, &ONE, &ONE, 
                pRPA->desc_Q_BLCYC, &beta, X_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC);
        if (pRPA->flagCyclicPermute) {
            double *X_BLCYC_permuted = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
            pdgemm_("N", "N", &DMndspe, &pRPA->nuChi0Neig, &pRPA->nuChi0Neig, &alpha, 
                    X_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, pRPA->permute_BLCYC, &ONE, &ONE, 
                    pRPA->desc_permute_BLCYC, &beta, X_BLCYC_permuted, &ONE, &ONE, pRPA->desc_orb_BLCYC);
            memcpy(X_BLCYC, X_BLCYC_permuted, pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
            free(X_BLCYC_permuted);
        }
    } else {
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            DMndspe, pRPA->nuChi0Neig, pRPA->nuChi0Neig, 
            1.0, nuChi0Y_BLCYC, DMndspe, Q, pRPA->nuChi0Neig,
            0.0, deltaVs, DMndspe
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
        pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, X_BLCYC, &ONE, &ONE, 
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
// #endif // (USE_MKL or USE_SCALAPACK)

    // if (printFlag) {
    //     int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;
    //     if (1){ // (pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)
    //         char afterFilterName[100];
    //         snprintf(afterFilterName, 100, "rank%d_nuChi0Eigscomm%d_deltaVs_afterCheFSI.txt", rank, pRPA->nuChi0EigscommIndex);
    //         FILE *outputYs = fopen(afterFilterName, "a");
    //         if (outputYs ==  NULL) {
    //             printf("error printing deltaVs_afterCheFSI\n");
    //             exit(EXIT_FAILURE);
    //         } else {
    //             for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //                 for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
    //                     fprintf(outputYs, "%12.9f ", rotatedEigVecs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
    //                 }
    //                 fprintf(outputYs, "\n");
    //             }
    //             fprintf(outputYs, "\n");
    //         }
    //         fclose(outputYs);
    //     }
    // }
}
