/**
 * @file    subspaceIter.c
 * @brief   This file contains the function controlling the subspace iteration framework with eigensolver.
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

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "restoreElectronicGroundState.h"
#include "subspaceIter.h"
#include "collectOrbitals.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "eigenSolverKpt_RPA.h"
#include "printResult.h"
#include "tools_RPA.h"

void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (flagNoDmcomm) return;
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int Nd = pSPARC->Nx * pSPARC->Ny * pSPARC->Nz;
    int seed_offset = 0;
    double vec2norm = 0.0;
    // seeds make sure that the initial deltaVs are identical in all processors of nuChi0Eigscomm. Unnecessary to broadcast.
    if (pSPARC->isGammaPoint) {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
                SeededRandVec(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
                Vector2Norm(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
                VectorScale(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
        }
    }
    else {
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
                SeededRandVec_complex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, gridsizes, -0.5, 0.5, seed_offset + Nd * (pRPA->nuChi0EigsStartIndex + i)); // deltaVs vectors are not normalized.
                Vector2Norm_complex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
                VectorScaleComplex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
        }
    }
}


void subspace_iteration_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int outputFirstEigAmount = (pRPA->nuChi0Neig > 2) ? 2 : pRPA->nuChi0Neig;
    int outputLastEigAmount = (pRPA->nuChi0Neig > 2) ? 2 : pRPA->nuChi0Neig;
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
        FILE *output_fp = fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"q-point %d (reduced coords %.3f %.3f %.3f), weight %.3f\n omega %d (value %.3f, 0~1 value %.3f, weight %.3f)\n",
            qptIndex + 1, pRPA->q1[qptIndex]*pSPARC->range_x/(2*M_PI), pRPA->q2[qptIndex]*pSPARC->range_y/(2*M_PI), pRPA->q3[qptIndex]*pSPARC->range_z/(2*M_PI), pRPA->qptWts[qptIndex],
            omegaIndex + 1, pRPA->omega[omegaIndex], pRPA->omega01[omegaIndex], pRPA->omegaWts[omegaIndex]);
        fprintf(output_fp,"nIter|ErpaTerm(Ha/atom)|First %d eigs & Last %d eigs of nu chi0|Timing (s)\n", outputFirstEigAmount, outputLastEigAmount);
        fclose(output_fp);
    }

    #ifdef DEBUG
    initialize_timing_counter(pRPA);
    #endif

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);

    if (pRPA->flagCOCGinitial && (!flagNoDmcomm)) {
        if (pSPARC->isGammaPoint) {
            collect_allXorb_allLambdas_gamma(pSPARC, pRPA);
        } else {
            collect_allXorb_allLambdas_kpt(pSPARC, pRPA);
            send_recv_allXorb_allLambdas_kPq(pSPARC, pRPA, qptIndex);
        }
    }

    double t1 = 0.0, t2 = 0.0;
    double lastErpaTerm = 10000.0, ErpaTerm = 0.0;
    double qptOmegaWeight = pRPA->qptWts[qptIndex] / pSPARC->Nkpts_sym * pRPA->omegaWts[omegaIndex]; // pSPARC->Nkpts_sym at here is total number of kpts (no sym)
    double tolErpaTermConverge = pRPA->tol_ErpaConverge * (double)pSPARC->n_atom / (double)pRPA->Nomega;
    
    int flagIter = 1;
    int signImag = 0;
    int printFlag = 0; // for midvariables output
    double t3, t4;
    for (int nIter = 1; nIter < pRPA->maxitFiltering + 1; nIter++) {
        t1 = MPI_Wtime();
        if (pSPARC->isGammaPoint) {
            project_tildeChi_general_eigProblem(pSPARC, pRPA, rank, nuChi0EigscommIndex, omegaIndex, flagNoDmcomm, nIter);
        } else {
            project_tildeChi_general_eigProblem_kpt(pSPARC, pRPA, rank, nuChi0EigscommIndex, qptIndex, omegaIndex, flagNoDmcomm, nIter);
        }

        if (!pRPA->nuChi0EigscommIndex) { // rank 0 in a nuChi0Eigscomm must be in dmcomm_phi of this nuChi0Eigscomm
        // if pRPA->nuChi0EigscommIndex == 0, then all processors in its dmcomm_phi contain all the eigenvalues
            if (!rank) { // to make things simple, just ask rank 0 of the 0th nuChi0Eigscomm to compute convergence
                ErpaTerm = compute_ErpaTerm(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, pRPA->omega01[omegaIndex], qptOmegaWeight);
                flagIter = fabs(ErpaTerm - lastErpaTerm) > tolErpaTermConverge ? 1 : 0;
                printf("qpt %d, omega %d, nIter %d, ErpaTerm %.6f, lastErpaTerm %.6f, tolErpaTermConverge %.6f, flagIter %d\n", qptIndex, omegaIndex, nIter, ErpaTerm, lastErpaTerm, tolErpaTermConverge, flagIter);
                lastErpaTerm = ErpaTerm;
            }
            MPI_Bcast(&flagIter, 1, MPI_INT, 0, pRPA->nuChi0Eigscomm);
        }
        MPI_Bcast(&flagIter, 1, MPI_INT, 0, pRPA->nuChi0EigsBridgeComm); // broadcast the flag to all processors of all nuChi0Eigscomms
        t2 = MPI_Wtime();

        if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
            FILE *output_fp = fopen(pRPA->filename_out,"a");
            fprintf(output_fp,"% 3d   %.8E  ", nIter, ErpaTerm / (double)pSPARC->n_atom);
            for (int eigIndex = 0; eigIndex < outputFirstEigAmount; eigIndex++) {
                fprintf(output_fp, "%.5f ", pRPA->RRnuChi0Eigs[eigIndex]);
            }
            fprintf(output_fp, "; ");
            for (int eigIndex = pRPA->nuChi0Neig - outputLastEigAmount; eigIndex < pRPA->nuChi0Neig; eigIndex++) {
                fprintf(output_fp, "%.5f ", pRPA->RRnuChi0Eigs[eigIndex]);
            }
            fprintf(output_fp, " %.2f\n", t2 - t1);
            fclose(output_fp);
        }
        if (flagIter == 0) break;
        if (nIter == 1) {
            printFlag = 0;
        }
        MPI_Barrier(pRPA->nuChi0Eigscomm);
    }
    #ifdef DEBUG
    finalize_timing_counter(pRPA, qptIndex, omegaIndex, flagNoDmcomm);
    #endif
    pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex] = ErpaTerm;

    if (pRPA->printEigs) {
        print_eigs_nuchi0(pRPA, qptIndex, omegaIndex);
    }
}


void project_tildeChi_general_eigProblem(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int omegaIndex, int flagNoDmcomm, int nIter) {
    double t3, t4;
    double *midVector_BLCYC, *deltaVs_BLCYC, *Ys_BLCYC;
    if (pRPA->npnuChi0Neig > 1) {
        midVector_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
        deltaVs_BLCYC = midVector_BLCYC;
    } else {
        deltaVs_BLCYC = pRPA->deltaVs;
    }
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);

    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    YT_multiply_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->deltaVs, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, deltaVs_BLCYC, pRPA->Mp, 0);
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumYT_mul_YTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, YT_multiply_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, pRPA->deltaVs, pRPA->Ys, pRPA->nNuChi0Eigscomm, flagNoDmcomm, nIter, 1);
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumOperatorMulTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, nu^0.5 chi0 nu^0.5 multiplying Ys spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    if (!pRPA->nuChi0EigsBridgeCommIndex) {
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        if (pRPA->npnuChi0Neig > 1) {
            Ys_BLCYC = midVector_BLCYC;
        } else {
            Ys_BLCYC = pRPA->Ys;
        }
        project_YT_nuChi0_Y_gamma(pRPA, pSPARC->dmcomm, deltaVs_BLCYC, pRPA->Ys, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, pRPA->Hp, Ys_BLCYC, 0);
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumYT_operator_YTime += t4 - t3;
        if ((!rank) && (!nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, project_YT_operator_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif

        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        generalized_eigenproblem_solver_gamma(pRPA, pRPA->nuChi0BlacsComm, 
            pRPA->Hp, pRPA->Mp, pRPA->nr_Hp_BLCYC, pRPA->nc_Hp_BLCYC, pRPA->desc_Hp_BLCYC, pRPA->desc_Mp_BLCYC, 
            pRPA->RRnuChi0Eigs, pRPA->Q, pRPA->nr_Q_BLCYC, pRPA->nc_Q_BLCYC, pRPA->desc_Q_BLCYC,
            pSPARC->eig_paral_blksz, flagNoDmcomm, 0); // pSPARC->eig_paral_blksz
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumEigTime += t4 - t3;
        if ((!rank) && (!nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, generalized_eigenproblem_solver spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif

        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        subspace_rotation_update_deltaVs(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, 
            Ys_BLCYC, pRPA->deltaVs, flagNoDmcomm, 0); // to save a multiplication in the Chebyshev filtering of the next subspace iteration
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            double vec2norm;
            Vector2Norm(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
            VectorScale(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
        }
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumRotationTime += t4 - t3;
        if ((!rank) && (!nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, subspace_rotation spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif
        MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0BlacsComm);
    }
    MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
    MPI_Bcast(pRPA->deltaVs, pSPARC->Nd * pRPA->nNuChi0Eigscomm, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
    if (pRPA->npnuChi0Neig > 1) {
        free(midVector_BLCYC);
    }
}


void project_tildeChi_general_eigProblem_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int qptIndex, int omegaIndex, int flagNoDmcomm, int nIter) {
    double t3, t4;
    double _Complex *midVector_BLCYC, *deltaVs_kptBLCYC, *Ys_kptBLCYC;
    if (pRPA->npnuChi0Neig > 1) {
        midVector_BLCYC = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
        deltaVs_kptBLCYC = midVector_BLCYC;
    } else {
        deltaVs_kptBLCYC = pRPA->deltaVs_kpt;
    }
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);

    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    YT_multiply_Y_kpt(pRPA, pSPARC->dmcomm, pRPA->deltaVs_kpt, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, deltaVs_kptBLCYC, pRPA->Mp_kpt, 0);
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumYT_mul_YTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, YT_multiply_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    #ifdef DEBUG
    t3 = MPI_Wtime();
    #endif
    nuChi0_mult_vectors_kpt(pSPARC, pRPA, qptIndex, omegaIndex, pRPA->deltaVs_kpt, pRPA->Ys_kpt, pRPA->nNuChi0Eigscomm, flagNoDmcomm, nIter, 1); // deltaVs_kpt = (\nu\chi0)*Ys_kpt for saving memory
    MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
    #ifdef DEBUG
    t4 = MPI_Wtime();
    pRPA->sumOperatorMulTime += t4 - t3;
    if ((!rank) && (!nuChi0EigscommIndex)) {
        printf("omega %d, subspace iteration %d, nu^0.5 chi0 nu^0.5 multiplying Ys spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
    }
    #endif

    // if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
    //     FILE *outputdVs = fopen("deltaVs", "a");
    //     for (int dVindex = 0; dVindex < pRPA->nNuChi0Eigscomm; dVindex++) {
    //         int Nd = pSPARC->Nd_d_dmcomm;
    //         fprintf(outputdVs, "dV | Ys\n");
    //         for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //             fprintf(outputdVs, "%12.9f + %12.9fi %12.9f + %12.9fi\n", creal(pRPA->deltaVs_kpt[dVindex*Nd + index]), cimag(pRPA->deltaVs_kpt[dVindex*Nd + index]), 
    //                 creal(pRPA->Ys_kpt[dVindex*Nd + index]), cimag(pRPA->Ys_kpt[dVindex*Nd + index]));
    //         }
    //     }
    //     fclose(outputdVs);
    // }

    if (!pRPA->nuChi0EigsBridgeCommIndex) {
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        if (pRPA->npnuChi0Neig > 1) {
            Ys_kptBLCYC = midVector_BLCYC;
        } else {
            Ys_kptBLCYC = pRPA->Ys_kpt;
        }
        project_YT_nuChi0_Y_kpt(pRPA, pSPARC->dmcomm, deltaVs_kptBLCYC, pRPA->Ys_kpt, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, pRPA->Hp_kpt, Ys_kptBLCYC, 0);
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumYT_operator_YTime += t4 - t3;
        if ((!rank) && (!nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, project_YT_operator_Y spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif

        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        generalized_eigenproblem_solver_kpt(pRPA, pRPA->nuChi0BlacsComm, 
            pRPA->Hp_kpt, pRPA->Mp_kpt, pRPA->nr_Hp_BLCYC, pRPA->nc_Hp_BLCYC, pRPA->desc_Hp_BLCYC, pRPA->desc_Mp_BLCYC, 
            pRPA->RRnuChi0Eigs, pRPA->Q_kpt, pRPA->nr_Q_BLCYC, pRPA->nc_Q_BLCYC, pRPA->desc_Q_BLCYC,
            pSPARC->eig_paral_blksz, flagNoDmcomm, 0); // pSPARC->eig_paral_blksz
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumEigTime += t4 - t3;
        if ((!rank) && (!nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, generalized_eigenproblem_solver spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif

        #ifdef DEBUG
        t3 = MPI_Wtime();
        #endif
        subspace_rotation_update_deltaVs_kpt(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, 
            Ys_kptBLCYC, pRPA->deltaVs_kpt, flagNoDmcomm, 0); // to save a multiplication in the Chebyshev filtering of the next subspace iteration
        for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
            double vec2norm;
            Vector2Norm_complex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
            VectorScaleComplex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
        }
        MPI_Barrier(pRPA->nuChi0EigsBridgeComm);
        #ifdef DEBUG
        t4 = MPI_Wtime();
        pRPA->sumRotationTime += t4 - t3;
        if ((!rank) && (!nuChi0EigscommIndex)) {
            printf("omega %d, subspace iteration %d, subspace_rotation spent %.2E ms.\n", omegaIndex + 1, nIter, (t4 - t3)*1e3);
        }
        #endif
        MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0BlacsComm);
    }
    MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
    MPI_Bcast(pRPA->deltaVs_kpt, pSPARC->Nd * pRPA->nNuChi0Eigscomm, MPI_DOUBLE_COMPLEX, 0, pRPA->nuChi0Eigscomm);

    // if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
    //     FILE *outputdVs = fopen("new_deltaVs", "a");
    //     for (int dVindex = 0; dVindex < pRPA->nNuChi0Eigscomm; dVindex++) {
    //         int Nd = pSPARC->Nd_d_dmcomm;
    //         for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //             fprintf(outputdVs, "%12.9f + %12.9fi\n", creal(pRPA->deltaVs_kpt[dVindex*Nd + index]), cimag(pRPA->deltaVs_kpt[dVindex*Nd + index]));
    //         }
    //     }
    //     fclose(outputdVs);
    // }

    if (pRPA->npnuChi0Neig > 1) {
        free(midVector_BLCYC);
    }
}


double compute_ErpaTerm(double *RRnuChi0Eigs, int nuChi0Neig, double omegaMesh01, double qptOmegaWeight) {
    double ErpaTerm = 0.0;
    for (int i = 0; i < nuChi0Neig; i++) {
        ErpaTerm += log(1.0 - RRnuChi0Eigs[i]) + RRnuChi0Eigs[i];
    }
    ErpaTerm *= qptOmegaWeight / (omegaMesh01*omegaMesh01) / (2.0*M_PI);
    return ErpaTerm;
}

#ifdef DEBUG
void initialize_timing_counter(RPA_OBJ *pRPA) {
    // timing counter for sternheimer equation for every qpt and omega
    pRPA->sternheimerTimeRecorder[0] = 0.0; // sumIniGuessTime

    pRPA->sternheimerTimeRecorder[1] = 0.0; // sumCOCGTime
    pRPA->sternheimerTimeRecorder[2] = 0.0; // sumLhsfunTime
    pRPA->sternheimerTimeRecorder[3] = 0.0; // sumSolveMuTime
    pRPA->sternheimerTimeRecorder[4] = 0.0; // sumMultipleTime
    // timing counter for operator multiplication
    pRPA->sumSqrtNuMulVecsTime = 0.0;
    pRPA->sumSternheimerTime = 0.0;
    pRPA->sumCollectdRhoTime = 0.0;
    // timing counter for subspace iteration
    pRPA->sumOperatorMulTime = 0.0;
    pRPA->sumOperatorMulTime_nuChi0EigsComm = 0.0;
    pRPA->sumYT_mul_YTime = 0.0;
    pRPA->sumYT_operator_YTime = 0.0;
    if (pRPA->flagSQ||pRPA->flagLQ) {
        pRPA->sum_orthonormalTime = 0.0;
        pRPA->sumTransfer_HpTime = 0.0;
        pRPA->sumSpectral_quadratureTime = 0.0;
    } else {
        pRPA->sumEigTime = 0.0;
        pRPA->sumRotationTime = 0.0;
    }
}


void finalize_timing_counter(RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm) {
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    if (!flagNoDmcomm) {
        FILE *outputFile = fopen(pRPA->timeRecordname, "a");
        // operator multiplications
        fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, sumOperatorMulTime_nuChi0EigsComm %.3f s, composed of sumSqrtNuMulVecsTime %.3f ms, sumSternheimerTime %.3f ms, sumCollectdRhoTime %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, 
            pRPA->sumOperatorMulTime_nuChi0EigsComm, pRPA->sumSqrtNuMulVecsTime*1e3, pRPA->sumSternheimerTime*1e3, pRPA->sumCollectdRhoTime*1e3);
        // Sternheimer equations
        fprintf(outputFile, "In sumSternheimerTime, sumIniGuessTime %.3f ms; sumCOCGTime %.3f ms, composed of sumLhsfunTime %.3f ms, sumSolveMuTime %.3f ms, sumMultipleTime %.3f ms\n", 
            pRPA->sternheimerTimeRecorder[0]*1e3, pRPA->sternheimerTimeRecorder[1]*1e3, pRPA->sternheimerTimeRecorder[2]*1e3, pRPA->sternheimerTimeRecorder[3]*1e3, pRPA->sternheimerTimeRecorder[4]*1e3);
        fclose(outputFile);
    }

    if (pRPA->flagSQ||pRPA->flagLQ) { //SQ
        if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
            printf("omega %d, sumOperatorMulTime %.3f ms, sumYT_mul_YTime %.3f ms, sum_orthonormalTime %.3f ms, sumYT_operator_YTime %.3f ms, sumTransfer_HpTime %.3f ms, sumSpectral_quadratureTime %.3f ms\n",
             omegaIndex + 1, pRPA->sumOperatorMulTime*1e3, pRPA->sumYT_mul_YTime*1e3, pRPA->sum_orthonormalTime*1e3, pRPA->sumYT_operator_YTime*1e3, pRPA->sumTransfer_HpTime*1e3, pRPA->sumSpectral_quadratureTime*1e3);
        }
    } else { //eig
        if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
            printf("omega %d, sumOperatorMulTime %.3f ms, sumYT_mul_YTime %.3f ms, sumYT_operator_YTime %.3f ms, sumEigTime %.3f ms, sumRotationTime %.3f ms\n",
             omegaIndex + 1, pRPA->sumOperatorMulTime*1e3, pRPA->sumYT_mul_YTime*1e3, pRPA->sumYT_operator_YTime*1e3, pRPA->sumEigTime*1e3, pRPA->sumRotationTime*1e3);
        }
    }
}
#endif