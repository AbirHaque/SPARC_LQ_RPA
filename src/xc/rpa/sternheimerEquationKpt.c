/**
 * @file    sternheimerEquationKpt.c
 * @brief   This file contains the function composing Sternheimer equation for every (qpt, trialVec, kpt, band) set
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
#include "kroneckerLaplacian.h"

#include "restoreElectronicGroundState.h"
#include "linearSolvers.h"
#include "sternheimerEquationKpt.h"

void sternheimer_eq_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int nuChi0EigsAmount, double _Complex *DVs, int printFlag) { // compute \Delta\rho by solving Sternheimer equations in all pSPARC->dmcomm s
    #ifdef DEBUG
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double t1 = MPI_Wtime();
    #endif
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int Ns = pSPARC->Nstates;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;

    for (int index = 0; index < pSPARC->Nd_d_dmcomm * nuChi0EigsAmount; index++) {
        pRPA->deltaRhos_kpt[index] = 0.0;
    }
    if (nuChi0EigsAmount < 1) return; 

    double *poisC_invE_precond = pRPA->pois_const_precond;

    int bandStartIndex = pSPARC->band_start_indx;
    int bandEndIndex = pSPARC->band_end_indx;

    for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        for (int kpt = 0; kpt < Nkpts_kptcomm; kpt++) {
            int kg = kpt + pSPARC->kpt_start_indx;
            int kPq = pRPA->kPqList[kg][qptIndex];
            int kMq = pRPA->kMqList[kg][qptIndex];
            for (int bandIndex = bandStartIndex; bandIndex < bandEndIndex + 1; bandIndex++) { // every processor in a kpt comm has ALL lambda and occ of the k-point!
                // printf("kpt %d, band %d, epsilon %9.6f, occ %9.6f\n", kpt, bandIndex, pSPARC->lambda[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex], pSPARC->occ[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex]);
                if (pSPARC->occ[spn_i * Nkpts_kptcomm * Ns + kpt * Ns + bandIndex] < 1e-4) continue; // do not compute conduct band (blank band)
                double epsilon = pSPARC->lambda[spn_i * Nkpts_kptcomm * Ns + kpt * Ns + bandIndex];
                int localBandIndex = bandIndex - bandStartIndex;
                double _Complex *psi_kpt = pSPARC->Xorb_kpt + kpt * ncol * DMndsp + localBandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
                double bandWeight = pSPARC->occfac * (pSPARC->kptWts_loc[kpt] / pSPARC->Nkpts) * pSPARC->occ[spn_i * Nkpts_kptcomm * Ns + kpt * Ns + bandIndex];
                // char orbitalFileName[100];
                // snprintf(orbitalFileName, 100, "psi_kpt%d_band%d_spin%d.orbit", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                // FILE *outputPsi = fopen(orbitalFileName, "w");
                // if (outputPsi ==  NULL) {
                //     printf("error printing psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                //     exit(EXIT_FAILURE);
                // } else {
                //     for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                //         fprintf(outputPsi, "%12.9f %12.9f\n", creal(psi_kpt[index]), cimag(psi_kpt[index]));
                //     }
                // }
                // fclose(outputPsi);

                // compute constants for preconditioner

                double _Complex *deltaPsisPlusOmega = pRPA->deltaPsis_kpt;
                double _Complex *deltaPsisMinusOmega = pRPA->deltaPsis_kpt + pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm;
                double _Complex *allXorb_thekPq = NULL;
                double *allLambdas_thekPq = NULL;
                if (pRPA->flagCOCGinitial) {
                    allXorb_thekPq = pRPA->allXorb_kPq + (spn_i*(Nkpts_kptcomm*Ns) + kpt*Ns)*pSPARC->Nd_d_dmcomm;
                    allLambdas_thekPq = pRPA->allLambdas_kPq + spn_i*(Nkpts_kptcomm*Ns) + kpt*Ns;
                }
                sternheimer_solver_kpt(pSPARC, spn_i, kPq, kg, epsilon, pRPA->omega[omegaIndex], pRPA->flagPQ,
                    pRPA->flagCOCGinitial, allXorb_thekPq, allLambdas_thekPq, Ns, 
                    pRPA->flagPreconditioner[omegaIndex], poisC_invE_precond, pRPA->Icoeff[omegaIndex],
                    deltaPsisPlusOmega, deltaPsisMinusOmega, DVs, psi_kpt, bandWeight, pRPA->deltaRhos_kpt, nuChi0EigsAmount,
                    pRPA->sternRelativeResTol, pRPA->SternBlockSize[omegaIndex], printFlag, pRPA->timeRecordname, pRPA->sternheimerTimeRecorder);
                printf("spn_i %d, globalKpt %d, globalBandIndex %d, omegaIndex %d, +-omega\n", spn_i, kpt + pSPARC->kpt_start_indx, bandIndex + pSPARC->band_start_indx, omegaIndex);

                // if (bandIndex == 0) {
                //     char deltaOrbitalFileName[100];
                //     snprintf(deltaOrbitalFileName, 100, "Dpsi_spin%d_kpt%d_band%d_band_dV.orbit", pSPARC->spin_start_indx + spn_i, pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex);
                //     FILE *outputBandDVs = fopen(deltaOrbitalFileName, "w");
                //     snprintf(deltaOrbitalFileName, 100, "Dpsi_spin%d_kpt%d_band%d_+omega.orbit", pSPARC->spin_start_indx + spn_i, pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex);
                //     FILE *outputDpsiMinus = fopen(deltaOrbitalFileName, "w");
                //     snprintf(deltaOrbitalFileName, 100, "Dpsi_spin%d_kpt%d_band%d_-omega.orbit", pSPARC->spin_start_indx + spn_i, pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex);
                //     FILE *outputDpsiPlus = fopen(deltaOrbitalFileName, "w");
                //     if ((outputBandDVs == NULL) || (outputDpsiMinus ==  NULL) || (outputDpsiPlus ==  NULL)) {
                //         printf("error printing delta psi kpt %d, band %d, spin %d\n", pSPARC->kpt_start_indx + kpt, pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
                //         exit(EXIT_FAILURE);
                //     }
                //     for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                //         fprintf(outputBandDVs, "%12.9f+%12.9fi ", creal(psi_kpt[index]), cimag(psi_kpt[index]));
                //         for (int vec = 0; vec < pRPA->nNuChi0Eigscomm; vec++) { // print \Delta psi for the last \Delta V
                //             fprintf(outputBandDVs, "%12.9f+%12.9fi ", creal(DVs[vec*pSPARC->Nd_d_dmcomm + index]), cimag(DVs[vec*pSPARC->Nd_d_dmcomm + index]));
                //             fprintf(outputDpsiMinus, "%12.9f+%12.9fi ", creal(deltaPsisMinusOmega[vec*pSPARC->Nd_d_dmcomm + index]), cimag(deltaPsisMinusOmega[vec*pSPARC->Nd_d_dmcomm + index]));
                //             fprintf(outputDpsiPlus, "%12.9f+%12.9fi ", creal(deltaPsisPlusOmega[vec*pSPARC->Nd_d_dmcomm + index]), cimag(deltaPsisPlusOmega[vec*pSPARC->Nd_d_dmcomm + index]));
                //         }
                //         fprintf(outputBandDVs, "\n");
                //         fprintf(outputDpsiMinus, "\n");
                //         fprintf(outputDpsiPlus, "\n");
                //     }
                //     fclose(outputBandDVs);
                //     fclose(outputDpsiMinus);
                //     fclose(outputDpsiPlus);
                // }
            }
        }
    }

    #ifdef DEBUG
    double t2 = MPI_Wtime();
    if (!rank) printf("nuChi0Eigscomm %d, solve %d delta Vs for all bands, spent %.3f ms\n", pRPA->nuChi0EigscommIndex, nuChi0EigsAmount, (t2 - t1)*1e3);
    #endif
}

void sternheimer_solver_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPqIndex, int kptIndex, double epsilon, double omega, int flagPQ,
    int flagCOCGinitial, double _Complex *allXorb_thekPq, double *allLambdas_thekPq, int NusedOrbitalForInitGuess, 
    int flagPreconditioner, double *poisC_invE_precond, double Icoeff,
    double _Complex *deltaPsisPlusOmega, double _Complex *deltaPsisMinusOmega, double _Complex *deltaVs_kpt, double _Complex *psi_kpt, double bandWeight, double _Complex *deltaRhos_kpt, int nuChi0EigsAmounts,
    double sternRelativeResTol, int SternBlockSize, int printFlag, char *outputName, double *timeRecorder)
{
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

    void (*lhsfun)(SPARC_OBJ *, int, int, double, double, double _Complex *, double _Complex *, int) = Sternheimer_lhs_kpt;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double sqrtdV = sqrt(pSPARC->dV);
    double _Complex *SternheimerRhs_kpt = (double _Complex *)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) { // solve each Sternheimer equation one by one, unlike gamma-point case
        for (int i = 0; i < DMnd; i++) {
            SternheimerRhs_kpt[nuChi0EigsIndex*DMnd + i] = -deltaVs_kpt[nuChi0EigsIndex*DMnd + i]*(psi_kpt[i]/sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
    }

    #ifdef DEBUG
    double t1 = MPI_Wtime();
    #endif
    set_initial_guess_deltaPsis_kpt(pSPARC, epsilon, omega, flagPQ, flagCOCGinitial, NusedOrbitalForInitGuess, allXorb_thekPq, allLambdas_thekPq,
        SternheimerRhs_kpt, deltaPsisPlusOmega, deltaPsisMinusOmega, nuChi0EigsAmounts);
    #ifdef DEBUG
    double t2 = MPI_Wtime();
    #endif
    // if (globalRank == 0) {
    //     char initialGuessFileName[100];
    //     snprintf(initialGuessFileName, 100, "initial_deltaPsis_kpt%d_+omega.orbit", kptIndex);
    //     FILE *initialPsi = fopen(initialGuessFileName, "a");
    //     if (initialPsi ==  NULL) {
    //         printf("error printing initial delta psi\n");
    //         exit(EXIT_FAILURE);
    //     } else {
    //         for (int vector = 0; vector < nuChi0EigsAmounts; vector++) {
    //             fprintf(initialPsi, "epsilon %12.9f, RHS/+omega/-omega\n", epsilon);
    //             for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //                 fprintf(initialPsi, "%12.9f + %12.9fi, %12.9f + %12.9fi, %12.9f + %12.9fi\n", creal(SternheimerRhs_kpt[vector*DMnd + index]), cimag(SternheimerRhs_kpt[vector*DMnd + index]),
    //                     creal(deltaPsisPlusOmega[vector*DMnd + index]), cimag(deltaPsisPlusOmega[vector*DMnd + index]), 
    //                     creal(deltaPsisMinusOmega[vector*DMnd + index]), cimag(deltaPsisMinusOmega[vector*DMnd + index]));
    //             }
    //             fprintf(initialPsi, "\n");
    //         }
    //         fprintf(initialPsi, "------------\n");
            
    //         fprintf(initialPsi, "allLambdas_thekPq\n");
    //         for (int psi = 0; psi < pSPARC->Nstates; psi++) {
    //             fprintf(initialPsi, "%12.9f, ", allLambdas_thekPq[psi]);
    //         }
    //         fprintf(initialPsi, "\n");
    //         fprintf(initialPsi, "allXorb_thekPq\n");
    //         for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //             for (int psi = 0; psi < pSPARC->Nstates; psi++) {
    //                 fprintf(initialPsi, "%12.9f + %12.9fi, ", creal(allXorb_thekPq[psi*DMnd + index]), cimag(allXorb_thekPq[psi*DMnd + index]));
    //             }
    //             fprintf(initialPsi, "\n");
    //         }
    //     }
    //     fclose(initialPsi);
    // }

    int loopTime = nuChi0EigsAmounts / SternBlockSize;
    if (nuChi0EigsAmounts % SternBlockSize) loopTime++;
    int startIndex = 0;
    #ifdef DEBUG
    double t3 = MPI_Wtime();
    FILE *outputFile;
    if (printFlag) outputFile = fopen(outputName, "a");
    #endif
    double sumLhsfunTime = 0.0, sumSolveMuTime = 0.0, sumMultipleTime = 0.0;
    for (int loop = 0; loop < loopTime; loop++) {
        double t5 = MPI_Wtime();
        int blockSize = (nuChi0EigsAmounts - startIndex) > SternBlockSize ? SternBlockSize : (nuChi0EigsAmounts - startIndex);
        int maxInnerIter = 60;
        int maxOuterIter = 30;
        double _Complex *deltaPsisPlusOmegaLoop = deltaPsisPlusOmega + startIndex*DMnd;
        double _Complex *deltaPsisMinusOmegaLoop = deltaPsisMinusOmega + startIndex*DMnd;
        double _Complex *SternheimerRhsLoop = SternheimerRhs_kpt + startIndex*DMnd;

        double *resNormRecords = (double *)calloc(sizeof(double), 2*(maxOuterIter * (maxInnerIter + 1))); // record Ferbenious norm of residual vector sets, not norm of all vecs
        double RHS_frobnormsq = 0.0;
        for (int vecIndex = 0; vecIndex < blockSize; vecIndex++) {
            double theVecNorm;
            Vector2Norm_complex(SternheimerRhsLoop + vecIndex*DMnd, DMnd, &theVecNorm, pSPARC->dmcomm);
            RHS_frobnormsq += theVecNorm * theVecNorm;
        } // just for calculating RHS_frobnormsq, no need to save them

        double lhsfunTime = 0.0; double solveMuTime = 0.0; double multipleTime = 0.0;
        int iterTime = block_CRM(lhsfun,
            pSPARC, spn_i, kPqIndex, psi_kpt, epsilon, omega, flagPQ, deltaPsisPlusOmegaLoop, deltaPsisMinusOmegaLoop,
            SternheimerRhsLoop, blockSize,
            // int flagPreconditioner, void (*precond)(SPARC_OBJ *, double *, double *, int, double *, double), double *poisC_invE_precond, double Icoeff,
            RHS_frobnormsq, sternRelativeResTol, maxInnerIter, maxOuterIter, resNormRecords, &lhsfunTime, &solveMuTime, &multipleTime);

        // int iterTimePlus = AAR_sternheimer_kpt(lhsfun,
        // pSPARC, spn_i, kPqIndex, psi_kpt, epsilon, omega, flagPQ, deltaPsisPlusOmegaLoop,
        // SternheimerRhsLoop,
        // // void (*precond_fun)(SPARC_OBJ *,int,double,double *,double*,MPI_Comm), double c, 
        // RHS_frobnormsq, sternRelativeResTol, maxOuterIter * (maxInnerIter + 1), resNormRecords, &lhsfunTime, &solveMuTime, &multipleTime,
        // 0.01, 0.01, 7, 6);

        // int iterTimeMinus = AAR_sternheimer_kpt(lhsfun,
        // pSPARC, spn_i, kPqIndex, psi_kpt, epsilon, -omega, flagPQ, deltaPsisMinusOmegaLoop,
        // SternheimerRhsLoop,
        // // void (*precond_fun)(SPARC_OBJ *,int,double,double *,double*,MPI_Comm), double c, 
        // RHS_frobnormsq, sternRelativeResTol, maxOuterIter * (maxInnerIter + 1), resNormRecords + (maxOuterIter * (maxInnerIter + 1)), &lhsfunTime, &solveMuTime, &multipleTime,
        // 0.01, 0.01, 7, 6);

        double t6 = MPI_Wtime();
        #ifdef DEBUG
        double finalPlusOmegaResidFrobnormsq = resNormRecords[iterTime*2];
        double finalMinusOmegaResidFrobnormsq = resNormRecords[iterTime*2 + 1];
        // double finalPlusOmegaResidFrobnormsq = resNormRecords[iterTimePlus + 1];
        // double finalMinusOmegaResidFrobnormsq = resNormRecords[(maxOuterIter * (maxInnerIter + 1)) + iterTimeMinus + 1];
        // if (globalRank == 0) {
        //     char resNormFileName[100];
        //     snprintf(resNormFileName, 100, "resNormRecords_loop%d.log", loop);
        //     FILE *resNorm = fopen(resNormFileName, "a");
        //     if (resNorm ==  NULL) {
        //         printf("error printing initial delta psi\n");
        //         exit(EXIT_FAILURE);
        //     } else {
        //         for (int index = 0; index < iterTime + 1; index++) {
        //             fprintf(resNorm, "%12.9f %12.9f\n", resNormRecords[index*2], resNormRecords[index*2 + 1]);
        //         }
        //         fprintf(resNorm, "\n");
        //         fprintf(resNorm, "------------\n");
        //     }
        //     fclose(resNorm);
        // }
        if (printFlag) {
            fprintf(outputFile, "start index %d, block size %d, block CRM iterated for %d times; square of RHS_frobnorm %.6E, square of finalResidFrobnorm %.6E %.6E, time %.2f ms\n", 
               startIndex, blockSize, iterTime, RHS_frobnormsq, finalPlusOmegaResidFrobnormsq, finalMinusOmegaResidFrobnormsq, (t6 - t5)*1e3);
            if (iterTime == maxOuterIter*(maxInnerIter + 1) - 1) {
                fprintf(outputFile, "It terminated without converging to the desired tolerance.\n");
            }
            // fprintf(outputFile, "start index %d, block size %d, AAR iterated for %d %d times; square of RHS_frobnorm %.6E, square of finalResidFrobnorm %.6E %.6E, time %.2f ms\n", 
            //     startIndex, blockSize, iterTimePlus, iterTimeMinus, RHS_frobnormsq, finalPlusOmegaResidFrobnormsq, finalMinusOmegaResidFrobnormsq, (t6 - t5)*1e3);
            // if ((iterTimePlus == maxOuterIter*(maxInnerIter + 1) - 1) || (iterTimeMinus == maxOuterIter*(maxInnerIter + 1) - 1)) {
            //     fprintf(outputFile, "It terminated without converging to the desired tolerance.\n");
            // }
        }
        #endif
        startIndex += SternBlockSize;
        free(resNormRecords);
        sumLhsfunTime += lhsfunTime;
        sumSolveMuTime += solveMuTime;
        sumMultipleTime += multipleTime;
    }
    #ifdef DEBUG
    double t4 = MPI_Wtime();
    // if (printFlag) {
    //     fprintf(outputFile, "Here are %d blocks for all %d rhs, initial_guessTime %.3f ms, solver spent %.3f ms, sumLhsfunTime %.3f ms, sumSolveMuTime %.3f ms, sumMultipleTime %.3f ms\n", 
    //         loopTime, nuChi0EigsAmounts, (t2 - t1)*1e3, (t4 - t3)*1e3, sumLhsfunTime*1e3, sumSolveMuTime*1e3, sumMultipleTime*1e3);
    // }
    if (printFlag) fclose(outputFile);
    timeRecorder[0] += t2 - t1;
    timeRecorder[1] += t4 - t3;
    timeRecorder[2] += sumLhsfunTime;
    timeRecorder[3] += sumSolveMuTime;
    timeRecorder[4] += sumMultipleTime;
    #endif

    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) { // bandWeight includes occupation, kpt and spin factor
            deltaRhos_kpt[nuChi0EigsIndex * DMnd + i] += bandWeight * conj(psi_kpt[i] / sqrtdV) * (deltaPsisPlusOmega[nuChi0EigsIndex * DMnd + i] + deltaPsisMinusOmega[nuChi0EigsIndex * DMnd + i]); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
    }

    free(SternheimerRhs_kpt);
}

void Sternheimer_lhs_kpt(SPARC_OBJ *pSPARC, int spn_i, int kPq, double epsilon, double omega, double _Complex *X, double _Complex *lhsX, int nuChi0EigsAmounts)
{
    int sg = pSPARC->spin_start_indx + spn_i;
    int DMnd = pSPARC->Nd_d_dmcomm;
    // int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    pSPARC->k1_loc[0] = pSPARC->k1[kPq];
    pSPARC->k2_loc[0] = pSPARC->k2[kPq];
    pSPARC->k3_loc[0] = pSPARC->k3[kPq]; // to acommodate the function Lap_plus_diag_vec_mult_nonorth_kpt
    Hamiltonian_vectors_mult_kpt( // (Hamiltonian + c * I), use kpt function directly to operate complex variables
        pSPARC, DMnd, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, X, DMnd, lhsX, DMnd, spn_i, 0, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd*nuChi0EigsAmounts; i++)
    {
        lhsX[i] += omega * I * X[i];
    }
}

void set_initial_guess_deltaPsis_kpt(SPARC_OBJ *pSPARC, double epsilon, double omega, int flagPQ, int flagCOCGinitial, int NusedOrbital, double _Complex *allXorb_thekPq, double *allLambdas_thekPq,
                                 double _Complex *SternheimerRhs, double _Complex *deltaPsisPlusOmega, double _Complex *deltaPsisMinusOmega, int nuChi0EigsAmounts)
{
    int DMnd = pSPARC->Nd_d_dmcomm;
    int Nstates = pSPARC->Nstates;
    // going to add the code for generating the initial guess based on psis
    if (flagCOCGinitial) {
        double _Complex *diagMatrix = (double _Complex*)calloc(sizeof(double _Complex), NusedOrbital*NusedOrbital);
        double _Complex *midVar = (double _Complex*)calloc(sizeof(double _Complex), NusedOrbital*nuChi0EigsAmounts);
        double _Complex *midVar2 = (double _Complex*)calloc(sizeof(double _Complex), NusedOrbital*nuChi0EigsAmounts);
        
        for (int i = 0; i < NusedOrbital; i++) {
            diagMatrix[i*NusedOrbital + i] = 1.0 / (allLambdas_thekPq[i] + (double _Complex)flagPQ/pSPARC->dV - (double _Complex)epsilon - I*omega);
        }
        double _Complex Nalpha = 1.0; double _Complex Nbeta = 0.0;
        for (int i = 0; i < DMnd*Nstates; i++) {
            allXorb_thekPq[i] = conj(allXorb_thekPq[i]);
        }
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, NusedOrbital, nuChi0EigsAmounts, DMnd,
                  &Nalpha, allXorb_thekPq, DMnd,
                  SternheimerRhs, DMnd, &Nbeta,
                  midVar, NusedOrbital);
        for (int i = 0; i < DMnd*Nstates; i++) {
            allXorb_thekPq[i] = conj(allXorb_thekPq[i]);
        }
        // minus omega
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NusedOrbital, nuChi0EigsAmounts, NusedOrbital,
                  &Nalpha, diagMatrix, NusedOrbital,
                  midVar, NusedOrbital, &Nbeta,
                  midVar2, NusedOrbital);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, NusedOrbital,
                  &Nalpha, allXorb_thekPq, DMnd,
                  midVar2, NusedOrbital, &Nbeta,
                  deltaPsisMinusOmega, DMnd);
        // plus omega
        for (int i = 0; i < NusedOrbital; i++) {
            diagMatrix[i*NusedOrbital + i] = 1.0 / (allLambdas_thekPq[i] + (double _Complex)flagPQ/pSPARC->dV - (double _Complex)epsilon + I*omega);
        }
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NusedOrbital, nuChi0EigsAmounts, NusedOrbital,
                  &Nalpha, diagMatrix, NusedOrbital,
                  midVar, NusedOrbital, &Nbeta,
                  midVar2, NusedOrbital);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, NusedOrbital,
                  &Nalpha, allXorb_thekPq, DMnd,
                  midVar2, NusedOrbital, &Nbeta,
                  deltaPsisPlusOmega, DMnd);
        free(diagMatrix);
        free(midVar);
        free(midVar2);
    } else {
        for (int i = 0; i < nuChi0EigsAmounts * DMnd; i++) {
            deltaPsisMinusOmega[i] = 0.0; // (double)rand() / (double)RAND_MAX;
            deltaPsisPlusOmega[i] = 0.0; // (double)rand() / (double)RAND_MAX;
        }
    }
}