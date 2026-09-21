/**
 * @file    finalization_RPA.c
 * @brief   This file contains the finalization function for RPA calculation
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

#include "blacs.h"

#include "finalization.h"

#include "finalization_RPA.h"
#include "kroneckerLaplacian.h"

void finalize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    
    // free sym k-points
    free(pRPA->kptWts);
    free(pRPA->k1);
    free(pRPA->k2);
    free(pRPA->k3);
    // free q-points
    free(pRPA->qptWts);
    free(pRPA->q1);
    free(pRPA->q2);
    free(pRPA->q3);
    for (int nk = 0; nk < pSPARC->Nkpts_sym; nk++) {
        free(pRPA->kPqSymList[nk]);
        free(pRPA->kPqList[nk]);
        free(pRPA->kMqList[nk]);
    }
    free(pRPA->kPqSymList);
    free(pRPA->kPqList);
    free(pRPA->kMqList);
    // free omega
    free(pRPA->omega);
    free(pRPA->omega01);
    free(pRPA->omegaWts);
    // free eigs
    free(pRPA->RRnuChi0Eigs);
    if(!pRPA->flagLQ){
        free(pRPA->RRnuChi0EigVecs);
    }
    free(pRPA->ErpaTerms);
    // free communicators
    MPI_Comm_free(&pRPA->nuChi0Eigscomm);
    MPI_Comm_free(&pRPA->nuChi0EigsBridgeComm);
    MPI_Comm_free(&pRPA->nuChi0BlacsComm);
    if (pSPARC->isGammaPoint) {
        free(pRPA->permute);
        free(pRPA->Hp);
        free(pRPA->Mp);
        free(pRPA->Q);
        free(pRPA->permute_BLCYC);
    } else {
        free(pRPA->permute_kpt);
        free(pRPA->Hp_kpt);
        free(pRPA->Mp_kpt);
        free(pRPA->Q_kpt);
        free(pRPA->permute_BLCYC_kpt);
    }
    if (pRPA->flagSQ||pRPA->flagLQ) {
        if (pSPARC->isGammaPoint) {
            free(pRPA->HpSQ);
        } else {
            free(pRPA->HpSQ_kpt);
        }
        
        if (pRPA->HpDiagCommIndex != -1) {
            free(pRPA->diagFHps);
        }
        MPI_Comm_free(&pRPA->commWorld);
    }
    #if defined(USE_MKL) || defined(USE_SCALAPACK)
    Cblacs_gridexit(pRPA->ictxt_blacs);
    Cblacs_gridexit(pRPA->ictxt_blacs_topo);
    if (pRPA->flagSQ||pRPA->flagLQ) {
        Cblacs_gridexit(pRPA->ictxt_HpSQ);
        Cblacs_gridexit(pRPA->ctxtWorld);
    }
    #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pRPA->nuChi0EigscommIndex != -1) {
        if (pSPARC->isGammaPoint) {
            free(pRPA->deltaVs);
            free(pRPA->Ys);
        } else {
            free(pRPA->deltaVs_kpt);
            free(pRPA->Ys_kpt);
        }
        int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
        if (!flagNoDmcomm) {
            if (pSPARC->isGammaPoint) {
                free(pRPA->nearbyBandIndicesGamma);
                free(pRPA->neighborBandIndicesGamma); // free NULL pointer is fine
                free(pRPA->neighborBandsGamma);
                free(pRPA->allEpsilonsGamma);
                free(pRPA->deltaRhos);
                free(pRPA->sprtNuDeltaVs);
                free(pRPA->deltaPsisReal);
                free(pRPA->deltaPsisImag);

                free(pSPARC->pois_const);
                free_kron_Lap(pSPARC->kron_lap_exx[0]);
                free(pSPARC->kron_lap_exx[0]);
                if (pRPA->flagCOCGinitial) {
                    free(pRPA->allXorb);
                    free(pRPA->allLambdas);
                }

                if (pRPA->flagAppliedPrecond) {
                    if (pSPARC->BC == 2) {
                        free(pRPA->pois_const_precond);
                    } else {
                        free(pRPA->inv_eig_precond);
                    }
                }
            } else {
                free(pRPA->deltaRhos_kpt);
                free(pRPA->sprtNuDeltaVs_kpt);
                free(pRPA->deltaPsis_kpt);
                if (pRPA->flagCOCGinitial) {
                    free(pRPA->allXorb_kpt);
                    free(pRPA->allXorb_kPq);
                    free(pRPA->allLambdas);
                    free(pRPA->allLambdas_kPq);
                }

                free(pSPARC->pois_const);
                for (int qptIndex = 0; qptIndex < pRPA->Nqpts_sym; qptIndex++) {
                    free_kron_Lap(pSPARC->kron_lap_exx[qptIndex]);
                    free(pSPARC->kron_lap_exx[qptIndex]);
                }
            }
        }
        Free_scfvar(pSPARC);
    }
    free(pSPARC->kron_lap_exx);
}
