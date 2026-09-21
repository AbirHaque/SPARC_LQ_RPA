/**
 * @file    nuChi0VecRoutines.c
 * @brief   This file contains the function multiplying operator \Tilde{\chi} with trial vectors.
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

#include "restoreElectronicGroundState.h"
#include "sternheimerEquation.h"
#include "sternheimerEquationKpt.h"
#include "electrostatics_RPA.h"

void nuChi0_mult_vectors_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int omegaIndex, double *DVs, double *nuChi0DVs, int nuChi0EigsAmount, int flagNoDmcomm, int nIter, int printFlag) {
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG    
    double tstart, tend, t1, t2;
    tstart = MPI_Wtime();
    t1 = MPI_Wtime();
#endif
    double *sqrtNuDVs = pRPA->sprtNuDeltaVs;
    Calculate_sqrtNu_vecs_gamma(pSPARC, DVs, sqrtNuDVs, nuChi0EigsAmount, 0, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG    
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, 1st Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // FILE *outputFile;
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, 1st Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumSqrtNuMulVecsTime += t2 - t1;
#endif

    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    if (!flagNoDmcomm) {
        sternheimer_eq_gamma(pSPARC, pRPA, omegaIndex, nuChi0EigsAmount, sqrtNuDVs, printFlag);
    }
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG    
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, sternheimer_eq_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, sternheimer_eq_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumSternheimerTime += t2 - t1;
#endif

    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    collect_deltaRho_gamma(pSPARC, pRPA->deltaRhos, nuChi0EigsAmount, 0, pRPA->nuChi0EigscommIndex);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG  
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, collect_deltaRho_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, collect_deltaRho_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumCollectdRhoTime += t2 - t1;
#endif

    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    Calculate_sqrtNu_vecs_gamma(pSPARC, pRPA->deltaRhos, nuChi0DVs, nuChi0EigsAmount, 0, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG  
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, 2nd Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, 2nd Calculate_sqrtNu_vecs_gamma spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumSqrtNuMulVecsTime += t2 - t1;
    tend = MPI_Wtime();
    pRPA->sumOperatorMulTime_nuChi0EigsComm += tend - tstart;
#endif
}

void nuChi0_mult_vectors_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int qptIndex, int omegaIndex, double _Complex *DVs_kpt, double _Complex *nuChi0DVs_kpt, int nuChi0EigsAmount, int flagNoDmcomm, int nIter, int printFlag) {
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG    
    double tstart, tend, t1, t2;
    tstart = MPI_Wtime();
    t1 = MPI_Wtime();
#endif
    double _Complex *sprtNuDVs_kpt = pRPA->sprtNuDeltaVs_kpt;
    Calculate_sqrtNu_vecs_kpt(pSPARC, DVs_kpt, sprtNuDVs_kpt, qptIndex, nuChi0EigsAmount, 0, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG    
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, 1st Calculate_sqrtNu_vecs_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // FILE *outputFile;
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, 1st Calculate_sqrtNu_vecs_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumSqrtNuMulVecsTime += t2 - t1;
#endif

    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    if (!flagNoDmcomm) {
        sternheimer_eq_kpt(pSPARC, pRPA, qptIndex, omegaIndex, nuChi0EigsAmount, sprtNuDVs_kpt, printFlag);
    }
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG    
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, sternheimer_eq_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, sternheimer_eq_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumSternheimerTime += t2 - t1;
#endif
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    collect_deltaRho_kpt(pSPARC, pRPA->deltaRhos_kpt, nuChi0EigsAmount, 0, pRPA->nuChi0EigscommIndex);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG  
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, collect_deltaRho_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, collect_deltaRho_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumCollectdRhoTime += t2 - t1;
#endif
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    Calculate_sqrtNu_vecs_kpt(pSPARC, pRPA->deltaRhos_kpt, nuChi0DVs_kpt, qptIndex, nuChi0EigsAmount, 0, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
    MPI_Barrier(pRPA->nuChi0Eigscomm);
#ifdef DEBUG  
    t2 = MPI_Wtime();
    if ((!pRPA->nuChi0EigscommIndex) && (!rank)) printf("nuChi0Eigscomm %d, rank %d, omega %d, nIter %d, 2nd Calculate_sqrtNu_vecs_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, nIter, (t2 - t1)*1e3);
    // if (printFlag) {
    //     outputFile = fopen(pRPA->timeRecordname, "a");
    //     fprintf(outputFile, "nuChi0Eigscomm %d, rank %d, omega %d, 2nd Calculate_sqrtNu_vecs_kpt spent %.3f ms\n", pRPA->nuChi0EigscommIndex, rank, omegaIndex, (t2 - t1)*1e3);
    //     fclose(outputFile);
    // }
    pRPA->sumSqrtNuMulVecsTime += t2 - t1;
    tend = MPI_Wtime();
    pRPA->sumOperatorMulTime_nuChi0EigsComm += tend - tstart;
#endif
}