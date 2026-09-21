/**
 * @file    subspaceIterSQ.c
 * @brief   This file contains the function controlling the subspace iteration framework with SQ trace estimator.
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

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "restoreElectronicGroundState.h"
#include "subspaceIter.h"
#include "subspaceIterSQ.h"
#include "collectOrbitals.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "eigenSolverKpt_RPA.h"
#include "tools_RPA.h"
#include "traceSolverSQ.h"

void subspace_iteration_RPASQ(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex) {
    // if (!pSPARC->isGammaPoint) {
    //     printf("RPA kpt is under development\n");
    //     return;
    // }
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex; // in the SQ code, processors with nuChi0EigscommIndex == -1 
    // can also stay in this function; since they may join any diagcomm 
    // however, they should not join X*deltaV functions and deltaV'*X*deltaV projection
    int rank, globalrank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
        FILE *output_fp = fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"q-point %d (reduced coords %.3f %.3f %.3f), weight %.3f\n omega %d (value %.3f, 0~1 value %.3f, weight %.3f)\n",
            qptIndex + 1, pRPA->q1[qptIndex]*pSPARC->range_x/(2*M_PI), pRPA->q2[qptIndex]*pSPARC->range_y/(2*M_PI), pRPA->q3[qptIndex]*pSPARC->range_z/(2*M_PI), pRPA->qptWts[qptIndex],
            omegaIndex + 1, pRPA->omega[omegaIndex], pRPA->omega01[omegaIndex], pRPA->omegaWts[omegaIndex]);
        fprintf(output_fp,"nIter|ErpaTerm(Ha/atom)|Timing (s)\n");
        fclose(output_fp);
    }
    int rankHpDiagComm;
    MPI_Comm_rank(pRPA->HpDiagComm, &rankHpDiagComm);
    MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);

    #ifdef DEBUG
    initialize_timing_counter(pRPA);
    #endif

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    
    if ((nuChi0EigscommIndex != -1) && pRPA->flagCOCGinitial && (!flagNoDmcomm)) {
        if (pSPARC->isGammaPoint) {
            collect_allXorb_allLambdas_gamma(pSPARC, pRPA);
        } else {
            collect_allXorb_allLambdas_kpt(pSPARC, pRPA);
            send_recv_allXorb_allLambdas_kPq(pSPARC, pRPA, qptIndex);
        }
    }

    double t1 = 0.0, t2 = 0.0;
    double ErpaTerm = 0.0, lastErpaTerm = 10000.0;
    double qptOmegaWeight = pRPA->qptWts[qptIndex] / pSPARC->Nkpts_sym * pRPA->omegaWts[omegaIndex];
    double tolErpaTermConverge = pRPA->tol_ErpaConverge * (double)pSPARC->n_atom / (double)pRPA->Nomega;

    int flagIter = 1;
    int nIter = 1;
    int printFlag = 0; // for midvariables output
    double t3, t4, t5, t6;
    for (nIter = 1; nIter < pRPA->maxitFiltering + 1; nIter++) {
        t1 = MPI_Wtime();
        if (pSPARC->isGammaPoint) {
            #ifdef DEBUG
            t5 = MPI_Wtime();
            #endif
            if (nuChi0EigscommIndex != -1) {
                project_tildeChi_update_deltaV(pSPARC, pRPA, rank, nuChi0EigscommIndex, omegaIndex, flagNoDmcomm, nIter);
            }
            #ifdef DEBUG
            t6 = MPI_Wtime();
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, project the operator into subspace spent %.2E ms.\n", omegaIndex + 1, nIter, (t6 - t5)*1e3);
            }
            #endif

            MPI_Barrier(MPI_COMM_WORLD);
            #ifdef DEBUG
            t5 = MPI_Wtime();
            #endif
            transfer_Hp_HpDiagComms(pRPA->nuChi0Neig, pRPA->Hp, pRPA->desc_Hp_BLCYC, pRPA->HpSQ, pRPA->desc_HpSQ, pRPA->nrHpSQ, pRPA->ncHpSQ,
                pRPA->nuChi0EigsBridgeCommIndex, pRPA->HpDiagCommIndex, pRPA->ctxtWorld, pRPA->HpDiagBridgeComm, rankHpDiagComm, 0);
            MPI_Barrier(MPI_COMM_WORLD);
            #ifdef DEBUG
            t6 = MPI_Wtime();
            pRPA->sumTransfer_HpTime += t6 - t5;
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, transfer_Hp_HpDiagComms spent %.2E ms.\n", omegaIndex + 1, nIter, (t6 - t5)*1e3);
            }
            t5 = MPI_Wtime();
            #endif

            double SQtimeRecorder[3] = {0.0, 0.0, 0.0}; // LanczosTimeRecord, diagonalizeTimeRecord, reduceTimeRecord
            MPI_Barrier(MPI_COMM_WORLD);
            int HpPrintFlag = 0;
            ErpaTerm = SQ_trace_estimator(pRPA, pRPA->HpSQ, pRPA->desc_HpSQ, pRPA->HpDiagStartIndex, pRPA->HpDiagEndIndex,
                qptOmegaWeight, omegaIndex, pRPA->HpDiagComm, SQtimeRecorder, rankHpDiagComm, HpPrintFlag);
            
            #ifdef DEBUG
            t6 = MPI_Wtime();
            pRPA->sumSpectral_quadratureTime += t6 - t5;
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, spectral_quadrature_funRPA spent %.2E ms, Lanczos time %.2E ms, diagonalize time %.2E ms, reduce time %.2E ms.\n", 
                    omegaIndex + 1, nIter, (t6 - t5)*1e3, SQtimeRecorder[0]*1e3, SQtimeRecorder[1]*1e3, SQtimeRecorder[2]*1e3);
            }
            #endif
        } else {
            #ifdef DEBUG
            t5 = MPI_Wtime();
            #endif
            if (nuChi0EigscommIndex != -1) {
                project_tildeChi_update_deltaV_kpt(pSPARC, pRPA, rank, nuChi0EigscommIndex, qptIndex, omegaIndex, flagNoDmcomm, nIter);
            }
            #ifdef DEBUG
            t6 = MPI_Wtime();
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, project the operator into subspace spent %.2E ms.\n", omegaIndex + 1, nIter, (t6 - t5)*1e3);
            }
            #endif

            MPI_Barrier(MPI_COMM_WORLD);
            #ifdef DEBUG
            t5 = MPI_Wtime();
            #endif
            transfer_Hp_HpDiagComms_kpt(pRPA->nuChi0Neig, pRPA->Hp_kpt, pRPA->desc_Hp_BLCYC, pRPA->HpSQ_kpt, pRPA->desc_HpSQ, pRPA->nrHpSQ, pRPA->ncHpSQ,
                pRPA->nuChi0EigsBridgeCommIndex, pRPA->HpDiagCommIndex, pRPA->ctxtWorld, pRPA->HpDiagBridgeComm, rankHpDiagComm, 0);
            MPI_Barrier(MPI_COMM_WORLD);
            #ifdef DEBUG
            t6 = MPI_Wtime();
            pRPA->sumTransfer_HpTime += t6 - t5;
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, transfer_Hp_HpDiagComms spent %.2E ms.\n", omegaIndex + 1, nIter, (t6 - t5)*1e3);
            }
            t5 = MPI_Wtime();
            #endif
            
            double SQtimeRecorder[3] = {0.0, 0.0, 0.0}; // LanczosTimeRecord, diagonalizeTimeRecord, reduceTimeRecord
            MPI_Barrier(MPI_COMM_WORLD);
            int HpPrintFlag = 0;
            ErpaTerm = SQ_trace_estimator_kpt(pRPA, pRPA->HpSQ_kpt, pRPA->desc_HpSQ, pRPA->HpDiagStartIndex, pRPA->HpDiagEndIndex,
                qptOmegaWeight, omegaIndex, pRPA->HpDiagComm, SQtimeRecorder, rankHpDiagComm, HpPrintFlag);
            
            #ifdef DEBUG
            t6 = MPI_Wtime();
            pRPA->sumSpectral_quadratureTime += t6 - t5;
            if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
                printf("omega %d, subspace iteration %d, spectral_quadrature_funRPA spent %.2E ms, Lanczos time %.2E ms, diagonalize time %.2E ms, reduce time %.2E ms.\n", 
                    omegaIndex + 1, nIter, (t6 - t5)*1e3, SQtimeRecorder[0]*1e3, SQtimeRecorder[1]*1e3, SQtimeRecorder[2]*1e3);
            }
            #endif
        }
        t2 = MPI_Wtime();
        if (!globalrank) {
            flagIter = fabs(ErpaTerm - lastErpaTerm) > tolErpaTermConverge ? 1 : 0;
            printf("qpt %d, omega %d, subspace iteration %d, ErpaTerm %.6f, lastErpaTerm %.6f, tolErpaTermConverge %.6f, flagIter %d\n", qptIndex + 1, omegaIndex + 1, nIter, ErpaTerm, lastErpaTerm, tolErpaTermConverge, flagIter);
            lastErpaTerm = ErpaTerm;
            FILE *output_fp = fopen(pRPA->filename_out,"a");
            fprintf(output_fp,"% 3d   %.8E   %.2f\n", nIter, ErpaTerm / (double)pSPARC->n_atom, t2 - t1);
            fclose(output_fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("global rank %d, omega %d, before Bcase %dth iter, flagIter %d.\n", globalrank, omegaIndex+1, nIter, flagIter);
        int info = MPI_Bcast(&flagIter, 1, MPI_INT, 0, pRPA->commWorld); // when the MPI_Comm is MPI_COMM_WORLD, there may be error in for or while loop! I need to create a new comm, which contains all processors.
        // printf("global rank %d, omega %d, reached the end of %dth iter, flagIter %d, bcast info %d.\n", globalrank, omegaIndex+1, nIter, flagIter, info); 
        if (flagIter == 0) break;
        if (nIter == 1) {
            printFlag = 0;
        }
    }
    #ifdef DEBUG
    finalize_timing_counter(pRPA, qptIndex, omegaIndex, flagNoDmcomm);
    #endif
    pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex] = ErpaTerm;

}


void project_tildeChi_update_deltaV(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int omegaIndex, int flagNoDmcomm, int nIter) {
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    double t3, t4;
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, pRPA->deltaVs, pRPA->Ys, pRPA->nNuChi0Eigscomm, flagNoDmcomm, nIter, 1);
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumOperatorMulTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, nuChi0_mult_vectors_gamma 1 spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    double *midVector_BLCYC, *deltaVs_BLCYC, *Ys_BLCYC;
    if (pRPA->npnuChi0Neig > 1) {
        midVector_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
        deltaVs_BLCYC = midVector_BLCYC;
    }
    if (!flagNoDmcomm) {
        if (pRPA->npnuChi0Neig > 1) {
            int DMndspe = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_eig;
            int ONE = 1;
            pdgemr2d_(&DMndspe, &pRPA->nuChi0Neig, pRPA->deltaVs, &ONE, &ONE, pRPA->desc_orbitals,
                deltaVs_BLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &pRPA->ictxt_blacs); // pRPA->Ys_BLCYC at here is the block cyclicly distributed new dVs
        } else {
            deltaVs_BLCYC = pRPA->deltaVs;
        }
    }

    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, getting Ys_BLCYC vectors spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif
    if (pRPA->npnuChi0Neig > 1) {
        Ys_BLCYC = midVector_BLCYC;
    } else {
        Ys_BLCYC = pRPA->Ys;
    }
    if (!pRPA->nuChi0EigsBridgeCommIndex) {
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        project_YT_nuChi0_Y_gamma(pRPA, pSPARC->dmcomm, deltaVs_BLCYC, pRPA->Ys, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, pRPA->Hp, Ys_BLCYC, 0);
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumYT_operator_YTime += t4 - t3;
        if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, project_YT_operator_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif
    }

    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    YT_multiply_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->Ys, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, Ys_BLCYC, pRPA->Mp, 0);
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumYT_mul_YTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, YT_multiply_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    Y_orth_permute_gamma(pSPARC, pRPA, Ys_BLCYC, pRPA->Mp, pRPA->deltaVs, flagNoDmcomm, 0); // the new dV vectors are formed at here, Ys_BLCYC saves Ys = X*dVs in block cyclic form
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sum_orthonormalTime += t4 - t3;
    if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, orthonormalizing deltaV vectors spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    if (pRPA->npnuChi0Neig > 1) {
        free(midVector_BLCYC);
    }
}


void project_tildeChi_update_deltaV_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int qptIndex, int omegaIndex, int flagNoDmcomm, int nIter) {
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    double t3, t4;
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    nuChi0_mult_vectors_kpt(pSPARC, pRPA, qptIndex, omegaIndex, pRPA->deltaVs_kpt, pRPA->Ys_kpt, pRPA->nNuChi0Eigscomm, flagNoDmcomm, nIter, 1);
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumOperatorMulTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, nuChi0_mult_vectors_kpt 1 spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    double _Complex *midVector_BLCYC, *deltaVs_kptBLCYC, *Ys_kptBLCYC;
    if (pRPA->npnuChi0Neig > 1) {
        midVector_BLCYC = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
        deltaVs_kptBLCYC = midVector_BLCYC;
    }
    if (!flagNoDmcomm) {
        if (pRPA->npnuChi0Neig > 1) {
            int DMndspe = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_eig;
            int ONE = 1;
            pzgemr2d_(&DMndspe, &pRPA->nuChi0Neig, pRPA->deltaVs_kpt, &ONE, &ONE, pRPA->desc_orbitals,
                deltaVs_kptBLCYC, &ONE, &ONE, pRPA->desc_orb_BLCYC, &pRPA->ictxt_blacs); // pRPA->Ys_BLCYC at here is the block cyclicly distributed new dVs
        } else {
            deltaVs_kptBLCYC = pRPA->deltaVs_kpt;
        }
    }
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, getting Ys_kpt_BLCYC vectors spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif
    if (pRPA->npnuChi0Neig > 1) {
        Ys_kptBLCYC = midVector_BLCYC;
    } else {
        Ys_kptBLCYC = pRPA->Ys_kpt;
    }
    if (!pRPA->nuChi0EigsBridgeCommIndex) {
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        project_YT_nuChi0_Y_kpt(pRPA, pSPARC->dmcomm, deltaVs_kptBLCYC, pRPA->Ys_kpt, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, pRPA->Hp_kpt, Ys_kptBLCYC, 0);
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumYT_operator_YTime += t4 - t3;
        if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, project_YT_operator_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif
    }

    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    YT_multiply_Y_kpt(pRPA, pSPARC->dmcomm, pRPA->Ys_kpt, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, Ys_kptBLCYC, pRPA->Mp_kpt, 0);
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumYT_mul_YTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, YT_multiply_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    Y_orth_permute_kpt(pSPARC, pRPA, Ys_kptBLCYC, pRPA->Mp_kpt, pRPA->deltaVs_kpt, flagNoDmcomm, 0); // the new dV vectors are formed at here, Ys_BLCYC saves Ys = X*dVs in block cyclic form
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sum_orthonormalTime += t4 - t3;
    if ((!rank) && (!pRPA->nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, orthonormalizing deltaV vectors spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    if (pRPA->npnuChi0Neig > 1) {
        free(midVector_BLCYC);
    }
}


double SQ_trace_estimator(RPA_OBJ *pRPA, double *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex,
    double qptOmegaWeight, int omegaIndex, MPI_Comm HpDiagComm, double *SQtimeRecorder, int rankHpDiagComm, int HpPrintFlag) {
    double localErpaTerm = 0.0;
    double ErpaTerm = 0.0;
    double t3, t4;
    if (pRPA->HpDiagCommIndex != -1) {
        double *TdiagArrayGroup = (double*)calloc(sizeof(double), pRPA->nHpDiagComm*pRPA->nplLanczosSQ);
        double *ToffDiagArrayGroup = (double*)calloc(sizeof(double), pRPA->nHpDiagComm*(pRPA->nplLanczosSQ - 1));
        double *TdiagArray = (double*)calloc(sizeof(double), pRPA->nplLanczosSQ);
        double *ToffDiagArray = (double*)calloc(sizeof(double), pRPA->nplLanczosSQ - 1);
        double *TeigVals = (double*)calloc(sizeof(double), pRPA->nplLanczosSQ);
        double *TeigVecs = (double*)calloc(sizeof(double), pRPA->nplLanczosSQ * pRPA->nplLanczosSQ); // also used for saving T matrix
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        // int HpPrintFlag = (diagIndex == pRPA->HpDiagStartIndex);
        int finalNpl = Lanczos_decompose_Hp_group(pRPA->nuChi0Neig, HpSQ, desc_HpSQ, HpDiagStartIndex, HpDiagEndIndex,
            pRPA->vkStart, pRPA->vkEnd, pRPA->nrvk, pRPA->desc_vk, pRPA->nrvk_BLCYC, pRPA->desc_vk_BLCYC, &pRPA->ictxt_HpSQ,
            pRPA->nplLanczosSQ, TdiagArrayGroup, ToffDiagArrayGroup, HpDiagComm, pRPA->HpDiagCommIndex, HpPrintFlag);
        MPI_Barrier(pRPA->HpDiagComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        SQtimeRecorder[0] += t4 - t3;
        t3 = MPI_Wtime();
        #endif
        for (int diagIndex = HpDiagStartIndex; diagIndex < HpDiagEndIndex + 1; diagIndex++) {
            int localDiagIndex = diagIndex - pRPA->HpDiagStartIndex;
            for (int order = 0; order < pRPA->nplLanczosSQ - 1; order++) {
                TdiagArray[order] = TdiagArrayGroup[order*pRPA->nHpDiagComm + localDiagIndex];
                ToffDiagArray[order] = ToffDiagArrayGroup[order*pRPA->nHpDiagComm + localDiagIndex];
            }
            TdiagArray[pRPA->nplLanczosSQ - 1] = TdiagArrayGroup[(pRPA->nplLanczosSQ - 1)*pRPA->nHpDiagComm + localDiagIndex];
            pRPA->diagFHps[diagIndex - pRPA->HpDiagStartIndex] = diagonalize_T_get_diag_fHp(finalNpl, diagIndex, TdiagArray, ToffDiagArray, TeigVals, TeigVecs,
                rankHpDiagComm, pRPA->HpDiagCommIndex, HpPrintFlag);
            localErpaTerm += pRPA->diagFHps[diagIndex - pRPA->HpDiagStartIndex];
        }
        MPI_Barrier(pRPA->HpDiagBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        SQtimeRecorder[1] += t4 - t3;
        t3 = MPI_Wtime();
        #endif
        MPI_Allreduce(&localErpaTerm, &ErpaTerm, 1, MPI_DOUBLE, MPI_SUM, pRPA->HpDiagBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        SQtimeRecorder[2] += t4 - t3;
        #endif
        ErpaTerm *= qptOmegaWeight / (pRPA->omega01[omegaIndex]*pRPA->omega01[omegaIndex]) / (2.0*M_PI);
        free(TdiagArrayGroup);
        free(ToffDiagArrayGroup);
        free(TdiagArray);
        free(ToffDiagArray);
        free(TeigVals);
        free(TeigVecs);
    }
    return ErpaTerm;
}


double SQ_trace_estimator_kpt(RPA_OBJ *pRPA, double _Complex *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex,
    double qptOmegaWeight, int omegaIndex, MPI_Comm HpDiagComm, double *SQtimeRecorder, int rankHpDiagComm, int HpPrintFlag) {
    double localErpaTerm = 0.0;
    double ErpaTerm = 0.0;
    double t3, t4;
    MPI_Barrier(MPI_COMM_WORLD);
    if (pRPA->HpDiagCommIndex != -1) {
        double _Complex *TdiagArrayGroup = (double _Complex*)calloc(sizeof(double _Complex), pRPA->nHpDiagComm*pRPA->nplLanczosSQ);
        double _Complex *ToffDiagArrayGroup = (double _Complex*)calloc(sizeof(double _Complex), pRPA->nHpDiagComm*(pRPA->nplLanczosSQ - 1));
        double _Complex *TdiagArray = (double _Complex*)calloc(sizeof(double _Complex), pRPA->nplLanczosSQ);
        double _Complex *ToffDiagArray = (double _Complex*)calloc(sizeof(double _Complex), pRPA->nplLanczosSQ - 1);
        double *TeigVals = (double*)calloc(sizeof(double), pRPA->nplLanczosSQ);
        double _Complex *TeigVecs = (double _Complex*)calloc(sizeof(double _Complex), pRPA->nplLanczosSQ * pRPA->nplLanczosSQ); // also used for saving T matrix
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        // int HpPrintFlag = (diagIndex == pRPA->HpDiagStartIndex);
        int HpPrintFlag = 0;
        int finalNpl = Lanczos_decompose_Hp_group_kpt(pRPA->nuChi0Neig, pRPA->HpSQ_kpt, pRPA->desc_HpSQ, pRPA->HpDiagStartIndex, pRPA->HpDiagEndIndex,
            pRPA->vkStart, pRPA->vkEnd, pRPA->nrvk, pRPA->desc_vk, pRPA->nrvk_BLCYC, pRPA->desc_vk_BLCYC, &pRPA->ictxt_HpSQ,
            pRPA->nplLanczosSQ, TdiagArrayGroup, ToffDiagArrayGroup, pRPA->HpDiagComm, pRPA->HpDiagCommIndex, HpPrintFlag);
        MPI_Barrier(pRPA->HpDiagComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        SQtimeRecorder[0] += t4 - t3;
        t3 = MPI_Wtime();
        #endif
        for (int diagIndex = pRPA->HpDiagStartIndex; diagIndex < pRPA->HpDiagEndIndex + 1; diagIndex++) {
            int localDiagIndex = diagIndex - pRPA->HpDiagStartIndex;
            for (int order = 0; order < pRPA->nplLanczosSQ - 1; order++) {
                TdiagArray[order] = TdiagArrayGroup[order*pRPA->nHpDiagComm + localDiagIndex];
                ToffDiagArray[order] = ToffDiagArrayGroup[order*pRPA->nHpDiagComm + localDiagIndex];
            }
            TdiagArray[pRPA->nplLanczosSQ - 1] = TdiagArrayGroup[(pRPA->nplLanczosSQ - 1)*pRPA->nHpDiagComm + localDiagIndex];
            pRPA->diagFHps[diagIndex - pRPA->HpDiagStartIndex] = diagonalize_T_get_diag_fHp_kpt(finalNpl, diagIndex, TdiagArray, ToffDiagArray, TeigVals, TeigVecs,
                rankHpDiagComm, pRPA->HpDiagCommIndex, HpPrintFlag);
            localErpaTerm += pRPA->diagFHps[diagIndex - pRPA->HpDiagStartIndex];
        }
        MPI_Barrier(pRPA->HpDiagBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        SQtimeRecorder[1] += t4 - t3;
        t3 = MPI_Wtime();
        #endif
        MPI_Allreduce(&localErpaTerm, &ErpaTerm, 1, MPI_DOUBLE, MPI_SUM, pRPA->HpDiagBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        SQtimeRecorder[2] += t4 - t3;
        #endif
        ErpaTerm *= qptOmegaWeight / (pRPA->omega01[omegaIndex]*pRPA->omega01[omegaIndex]) / (2.0*M_PI);
        free(TdiagArrayGroup);
        free(ToffDiagArrayGroup);
        free(TdiagArray);
        free(ToffDiagArray);
        free(TeigVals);
        free(TeigVecs);
    }
    return ErpaTerm;
}