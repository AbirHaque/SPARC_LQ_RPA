/**
 * @file    sternheimerEquation.c
 * @brief   This file contains the function composing Sternheimer equation for every (trialVec, band) set
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
#include "sternheimerEquation.h"

void sternheimer_eq_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, int nuChi0EigsAmount, double *DVs, int printFlag) // compute \Delta\rho by solving Sternheimer equations in all pSPARC->dmcomm s
{
    #ifdef DEBUG
    int rank;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    double t1 = MPI_Wtime();
    #endif
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int Ns = pSPARC->Nstates;
    int ncol = pSPARC->Nband_bandcomm;

    // double *sternSolverAccuracy = (double *)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm); // the sum of 2-norm of residuals of all Sternheimer eq.s assigned in this processor
    for (int index = 0; index < pSPARC->Nd_d_dmcomm * nuChi0EigsAmount; index++) {
        pRPA->deltaRhos[index] = 0.0;
    }
    if (nuChi0EigsAmount < 1) return; // it is possible that some nuChi0Eigscomms are not assigned eigenpairs. For example: distribute 1500 into 120 nuChi0Eigscomm, then the last 4 comms will be left blank

    double *poisC_invE_precond;
    if (pSPARC->BC == 2) {
        poisC_invE_precond = pRPA->pois_const_precond;
    } else {
        poisC_invE_precond = pRPA->inv_eig_precond;
    }

    int bandStartIndex = pSPARC->band_start_indx;
    int bandEndIndex = pSPARC->band_end_indx;

    for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) {
        for (int bandIndex = bandStartIndex; bandIndex < bandEndIndex + 1; bandIndex++) { // every processor in a kpt comm has ALL lambda and occ of the k-point!
            if (pSPARC->occ[spn_i * Ns + bandIndex] < 1e-4) continue; // do not compute conduct band (blank band)
            double epsilon = pSPARC->lambda[spn_i * Ns + bandIndex];
            //printf("GGGGGGGGGGGGGGGG %f (%d/%d)\n",epsilon, bandIndex,bandEndIndex + 1);
            int localBandIndex = bandIndex - bandStartIndex;
            double *psi = pSPARC->Xorb + localBandIndex * DMndsp + spn_i * pSPARC->Nd_d_dmcomm;
            double bandWeight = pSPARC->occfac * pSPARC->occ[spn_i * Ns + bandIndex]; // occfac contains spin factor
            // if (printFlag) {
            //     char orbitalFileName[100];
            //     snprintf(orbitalFileName, 100, "psi_band%d_spin%d.orbit", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //     FILE *outputPsi = fopen(orbitalFileName, "w");
            //     if (outputPsi ==  NULL) {
            //         printf("error printing psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //         exit(EXIT_FAILURE);
            //     } else {
            //         for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
            //             fprintf(outputPsi, "%12.9f\n", psi[index]);
            //         }
            //     }
            //     fclose(outputPsi);
            // }
            
            if (pRPA->flagPreconditioner[omegaIndex]) {
                compute_pois_kron_cons_LapPrecond(pRPA, pSPARC, epsilon);
            }
            
            double *allXorb_thes = pRPA->allXorb + spn_i*Ns * pSPARC->Nd_d_dmcomm;
            double *allLambdas_thes = pSPARC->lambda + spn_i*Ns; // every processor in a kpt comm has ALL lambda and occ of the k-point!
            sternheimer_solver_gamma(pSPARC, spn_i, epsilon, pRPA->omega[omegaIndex], pRPA->flagPQ, 
                pRPA->flagCOCGinitial, allXorb_thes, allLambdas_thes, Ns, 
                pRPA->flagPreconditioner[omegaIndex], poisC_invE_precond, pRPA->Icoeff[omegaIndex],
                pRPA->deltaPsisReal, pRPA->deltaPsisImag, DVs, psi, bandWeight, pRPA->deltaRhos, nuChi0EigsAmount,
                pRPA->sternRelativeResTol, pRPA->SternBlockSize[omegaIndex], printFlag, pRPA->timeRecordname, pRPA->sternheimerTimeRecorder); // sternSolverAccuracy[spn_i * ncol + bandIndex] = 
            // printf("spn_i %d, globalBandIndex %d, omegaIndex %d, stern res norm %.6E\n", spn_i, bandIndex + pSPARC->band_start_indx, omegaIndex, sternSolverAccuracy[spn_i * ncol + bandIndex]);
            
            // print the \Delta \psi vector from the first \Delta V.
            // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
            // if (printFlag) {
            //     char deltaOrbitalFileName[100];
            //     snprintf(deltaOrbitalFileName, 100, "Dpsi_band%d_spin%d.orbit", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //     FILE *outputDpsi = fopen(deltaOrbitalFileName, "w");
            //     if (outputDpsi ==  NULL) {
            //         printf("error printing delta psi band %d, spin %d\n", pSPARC->band_start_indx + bandIndex, pSPARC->spin_start_indx + spn_i);
            //         exit(EXIT_FAILURE);
            //     } else {
            //         for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
            //             for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
            //                 fprintf(outputDpsi, "%12.9f %12.9f  ", pRPA->deltaPsisReal[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index], pRPA->deltaPsisImag[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
            //             }
            //             fprintf(outputDpsi, "\n");
            //         }
            //     }
            //     fclose(outputDpsi);
            // }
        }
    }
    // free(sternSolverAccuracy);

    #ifdef DEBUG
    double t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, solve %d delta Vs for all bands, spent %.3f ms\n", pRPA->nuChi0EigscommIndex, nuChi0EigsAmount, (t2 - t1)*1e3);
    #endif
}


void sternheimer_solver_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, int flagPQ, 
                                int flagCOCGinitial, double *allXorb_thes, double *allLambdas_thes, int NusedOrbitalForInitGuess, 
                                int flagPreconditioner, double *poisC_invE_precond, double Icoeff,
                                double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, double bandWeight, double *deltaRhos, int nuChi0EigsAmounts,
                                double sternRelativeResTol, int SternBlockSize, int printFlag, char *outputName, double *timeRecorder)
{
    void (*lhsfun)(SPARC_OBJ *, int, double *, double, double, int, double *, double *, double _Complex *, int) = Sternheimer_lhs;
    void (*precond)(SPARC_OBJ *, double *, double *, int, double *, double) = laplace_preconditioner;
    int DMnd = pSPARC->Nd_d_dmcomm;
    double sqrtdV = sqrt(pSPARC->dV);
    double _Complex *SternheimerRhs = (double _Complex *)calloc(sizeof(double _Complex), DMnd*nuChi0EigsAmounts);
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) {
            SternheimerRhs[nuChi0EigsIndex*DMnd + i] = -deltaVs[nuChi0EigsIndex*DMnd + i]*(psi[i]/sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
        // linear operator P, only contains the term of psi to be computed, occ2 = occ
        if (flagPQ) {
            double _Complex dotProdPsiDeltaVPsi = 0.0;
            for (int i = 0; i < DMnd; i++) {
                dotProdPsiDeltaVPsi += -(psi[i])*SternheimerRhs[nuChi0EigsIndex*DMnd + i]; // don't add (psi[i]/sqrtdV)!
            }
            for (int i = 0; i < DMnd; i++) {
                SternheimerRhs[nuChi0EigsIndex*DMnd + i] += (psi[i])*dotProdPsiDeltaVPsi;
            }
        }
    }

    //Haque: Output Sternheimer RHS for analysis.
    /*char rhsFilename[256];
    sprintf(rhsFilename, "SternheimerRhs_omega_%.3f_rank_%d_first_LQ_spn_%d_.bin", omega, tmp_rank,spn_i);
    FILE *rhsFile = fopen(rhsFilename, "ab"); // Binary append mode to handle multiple calls safely
    if (rhsFile != NULL) {
        fwrite(SternheimerRhs, sizeof(double _Complex), DMnd * nuChi0EigsAmounts, rhsFile);
        fclose(rhsFile);
    } else {
        fprintf(stderr, "Error opening file %s for writing SternheimerRhs!\n", rhsFilename);
    }*/

    #ifdef DEBUG
    double t1 = MPI_Wtime();
    #endif
    int NusedOrbital = NusedOrbitalForInitGuess; // pRPA->NoccupiedOrbital or pSPARC->Nstates
    set_initial_guess_deltaPsis(pSPARC, epsilon, omega, flagPQ, flagCOCGinitial, NusedOrbital, allXorb_thes, allLambdas_thes,
        SternheimerRhs, deltaPsisReal, deltaPsisImag, nuChi0EigsAmounts);
    #ifdef DEBUG
    double t2 = MPI_Wtime();
    #endif

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
        int maxIter = pSPARC->Nd_d_dmcomm / blockSize;
        double *deltaPsisRealLoop = deltaPsisReal + startIndex*DMnd;
        double *deltaPsisImagLoop = deltaPsisImag + startIndex*DMnd;
        double _Complex *SternheimerRhsLoop = SternheimerRhs + startIndex*DMnd;

        double *resNormRecords = (double *)calloc(sizeof(double), (maxIter + 1) * blockSize); // 1000 is maximum iteration time
        double RHS_frobnormsq = 0.0;
        for (int vecIndex = 0; vecIndex < blockSize; vecIndex++) {
            Vector2Norm_complex(SternheimerRhsLoop + vecIndex*DMnd, DMnd, &(resNormRecords[vecIndex]), pSPARC->dmcomm);
            RHS_frobnormsq += resNormRecords[vecIndex]*resNormRecords[vecIndex];
        }

        double lhsfunTime = 0.0; double solveMuTime = 0.0; double multipleTime = 0.0;
        int iterTime = block_COCG(lhsfun, pSPARC, spn_i, psi, epsilon, omega, flagPQ, deltaPsisRealLoop, deltaPsisImagLoop, SternheimerRhsLoop, blockSize, 
            flagPreconditioner, precond, poisC_invE_precond, Icoeff,
            RHS_frobnormsq, sternRelativeResTol, maxIter, resNormRecords, &lhsfunTime, &solveMuTime, &multipleTime);
        
        double t6 = MPI_Wtime();
        #ifdef DEBUG
        double finalResidFrobnormsq = 0.0;
        for (int vecIndex = 0; vecIndex < blockSize; vecIndex++) {
            finalResidFrobnormsq += resNormRecords[blockSize*iterTime + vecIndex]*resNormRecords[blockSize*iterTime + vecIndex];
        }
        if (printFlag) {
            //fprintf(outputFile, "omega %.3f, start index %d, block size %d, block COCG iterated for %d times; square of RHS_frobnorm %.6E, square of finalResidFrobnorm %.6E, time %.2f ms\n",omega, startIndex, blockSize, iterTime, RHS_frobnormsq, finalResidFrobnormsq, (t6 - t5)*1e3);
            fprintf(outputFile, "%.3f, %d, %d, %d\n",omega, startIndex, blockSize, iterTime);
            if (iterTime == maxIter) {
                fprintf(outputFile, "It terminated without converging to the desired tolerance.\n");
            }
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
    if (printFlag) {
         //fprintf(outputFile, "omega %.3f: Here are %d blocks for all %d rhs, initial_guessTime %.3f ms, solver spent %.3f ms, sumLhsfunTime %.3f ms, sumSolveMuTime %.3f ms, sumMultipleTime %.3f ms\n", 
         //    omega, loopTime, nuChi0EigsAmounts, (t2 - t1)*1e3, (t4 - t3)*1e3, sumLhsfunTime*1e3, sumSolveMuTime*1e3, sumMultipleTime*1e3);
    }
    if (printFlag) fclose(outputFile);
    timeRecorder[0] += t2 - t1;
    timeRecorder[1] += t4 - t3;
    timeRecorder[2] += sumLhsfunTime;
    timeRecorder[3] += sumSolveMuTime;
    timeRecorder[4] += sumMultipleTime;
    #endif

    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
        for (int i = 0; i < DMnd; i++) { // bandWeight includes occupation and spin factor
            deltaRhos[nuChi0EigsIndex * DMnd + i] += 2 * bandWeight * deltaPsisReal[nuChi0EigsIndex * DMnd + i] * (psi[i] / sqrtdV); // the unit of \psi and \delta\psi in Sternheimer eq. are sqrt(e/V).
        }
    }

    free(SternheimerRhs);
}

void Sternheimer_lhs(SPARC_OBJ *pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *Xreal, double *Ximag, double _Complex *lhsX, int nuChi0EigsAmounts)
{
    int sg = pSPARC->spin_start_indx + spn_i;
    double dV = pSPARC->dV;
    int DMnd = pSPARC->Nd_d_dmcomm;
    // int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {
        lhsX[i] = 0.0;
    }
    double *lhsXreal_Xcomp = (double*)calloc(sizeof(double), DMnd * nuChi0EigsAmounts);
    Hamiltonian_vectors_mult( // (Hamiltonian - \epsilon * I)
        pSPARC, pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, Xreal, DMnd, lhsXreal_Xcomp, DMnd, spn_i, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {// sternheimer eq.s for all \Delta Vs are solved together
        lhsX[i] += lhsXreal_Xcomp[i] - omega*Xreal[i] * I;
    }
    Hamiltonian_vectors_mult( // (Hamiltonian - \epsilon * I)
        pSPARC, pSPARC->Nd_d_dmcomm, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
        pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, nuChi0EigsAmounts, -epsilon, Ximag, DMnd, lhsXreal_Xcomp, DMnd, spn_i, pSPARC->dmcomm); // reminder: ldi and ldo should not be DMndsp!
    for (int i = 0; i < DMnd * nuChi0EigsAmounts; i++) {// sternheimer eq.s for all \Delta Vs are solved together
        lhsX[i] += lhsXreal_Xcomp[i] * I + omega*Ximag[i];
    }
    // linear operator Q, only contains the term of psi to be computed, occ2 = occ
    if (flagPQ) {
        for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmounts; nuChi0EigsIndex++) {
            double dotProdPsiXreal = 0.0, dotProdPsiXimag = 0.0;
            for (int i = 0; i < DMnd; i++) {
                dotProdPsiXreal += psi[i]*Xreal[nuChi0EigsIndex*DMnd + i];
                dotProdPsiXimag += psi[i]*Ximag[nuChi0EigsIndex*DMnd + i];
            }
            for (int i = 0; i < DMnd; i++) {
                lhsX[nuChi0EigsIndex*DMnd + i] += psi[i] * (dotProdPsiXreal + dotProdPsiXimag * I) / dV;
            }
        }
    }
    free(lhsXreal_Xcomp);
}

void set_initial_guess_deltaPsis(SPARC_OBJ *pSPARC, double epsilon, double omega, int flagPQ, int flagCOCGinitial, int NusedOrbital, double *allXorb_thes, double *allLambdas_thes,
                                 double _Complex *SternheimerRhs, double *deltaPsisReal, double *deltaPsisImag, int nuChi0EigsAmounts)
{
    int DMnd = pSPARC->Nd_d_dmcomm;
    int Nstates = pSPARC->Nstates;
    // going to add the code for generating the initial guess based on psis
    if (flagCOCGinitial) {
        double _Complex *deltaPsis = (double _Complex*)calloc(sizeof(double _Complex), DMnd * nuChi0EigsAmounts);
        double _Complex *allXorbComp = (double _Complex*)calloc(sizeof(double _Complex), DMnd * NusedOrbital);
        double _Complex *diagMatrix = (double _Complex*)calloc(sizeof(double _Complex), NusedOrbital*NusedOrbital);
        double _Complex *midVar = (double _Complex*)calloc(sizeof(double _Complex), NusedOrbital*nuChi0EigsAmounts);
        double _Complex *midVar2 = (double _Complex*)calloc(sizeof(double _Complex), NusedOrbital*nuChi0EigsAmounts);
        for (int i = 0; i < NusedOrbital * DMnd; i++) {
            allXorbComp[i] = allXorb_thes[i];
        }
        for (int i = 0; i < NusedOrbital; i++) {
            diagMatrix[i*NusedOrbital + i] = 1.0 / (allLambdas_thes[i] + (double _Complex)flagPQ/pSPARC->dV - (double _Complex)epsilon - I*omega);
        }
        double _Complex Nalpha = 1.0; double _Complex Nbeta = 0.0;
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, NusedOrbital, nuChi0EigsAmounts, DMnd,
                  &Nalpha, allXorbComp, DMnd,
                  SternheimerRhs, DMnd, &Nbeta,
                  midVar, NusedOrbital);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NusedOrbital, nuChi0EigsAmounts, NusedOrbital,
                  &Nalpha, diagMatrix, NusedOrbital,
                  midVar, NusedOrbital, &Nbeta,
                  midVar2, NusedOrbital);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, DMnd, nuChi0EigsAmounts, NusedOrbital,
                  &Nalpha, allXorbComp, DMnd,
                  midVar2, NusedOrbital, &Nbeta,
                  deltaPsis, DMnd);
        for (int i = 0; i < nuChi0EigsAmounts * DMnd; i++) {
            deltaPsisReal[i] = creal(deltaPsis[i]);
            deltaPsisImag[i] = cimag(deltaPsis[i]);
        }
        free(deltaPsis);
        free(allXorbComp);
        free(diagMatrix);
        free(midVar);
        free(midVar2);
    } else {
        for (int i = 0; i < nuChi0EigsAmounts * DMnd; i++) {
            deltaPsisReal[i] = 0.0; // (double)rand() / (double)RAND_MAX;
            deltaPsisImag[i] = 0.0; // (double)rand() / (double)RAND_MAX;
        }
    }
}

void compute_pois_kron_cons_LapPrecond(RPA_OBJ *pRPA, SPARC_OBJ *pSPARC, double epsilon) {
    KRON_LAP* kron_lap = pSPARC->kron_lap_exx[0];

    if (pSPARC->BC == 2) { // periodic BC
        for (int i = 0; i < pSPARC->Nd; i++) {
            double G2 = -0.5*kron_lap->eig[i] - epsilon;
            if (fabs(G2) > 1e-4) {
                pRPA->pois_const_precond[i] = 1.0/G2; // Laplacian
            } else {
                // pRPA->pois_const_precond[i] = pSPARC->const_aux; // singular point
                pRPA->pois_const_precond[i] = 1e4; // singular point
            }
        }
    } else { // Dirichlet BC
        for (int i = 0; i < pSPARC->Nd; i++) {
            pRPA->inv_eig_precond[i] = 1.0 / (-0.5*(kron_lap->inv_eig[i]) * (kron_lap->inv_eig[i]) - epsilon); // in initialization.c, line 196, kron_lap->inv_eig[i] becomes sqrt(-eigL)
            if (fabs(pRPA->inv_eig_precond[i] < 1e-4)) {
                // pRPA->inv_eig_precond[i] = pSPARC->const_aux; // singular point
                pRPA->inv_eig_precond[i] = 1e4; // singular point
            }
        }
    }
}

void laplace_preconditioner(SPARC_OBJ *pSPARC, double *inputVecs, double *outputVecs, int vecsAmount, double *poisC_invE_precond, double Icoeff) {
    int Nd_d = pSPARC->Nd_d_dmcomm;
    KRON_LAP* kron_lap = pSPARC->kron_lap_exx[0];
    for (int vecIndex = 0; vecIndex < vecsAmount; vecIndex++) {
        Lap_Kron(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx, kron_lap->Vy, kron_lap->Vz,
                inputVecs + vecIndex*Nd_d, poisC_invE_precond, outputVecs + vecIndex*Nd_d);
    }

    double minMesh = (pSPARC->delta_x > pSPARC->delta_y) ? pSPARC->delta_y : pSPARC->delta_x;
    minMesh = (minMesh > pSPARC->delta_z) ? pSPARC->delta_z : minMesh;
    double IcoeffDivMesh = Icoeff / (minMesh*minMesh);
    for (int i = 0; i < vecsAmount*Nd_d; i++) {
        outputVecs[i] += IcoeffDivMesh * inputVecs[i];
    }
}

