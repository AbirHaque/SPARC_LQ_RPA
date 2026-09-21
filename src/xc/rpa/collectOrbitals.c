/**
 * @file    collectOrbitals.c
 * @brief   This file saves functions for collecting orbitals from each band comm to generate initial guess for
 *          Sternheimer equation solver. The function transferring the orbitals between k-point comms to handle
 *          k+q of Hamiltonian is also here.
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

#include "collectOrbitals.h"

void collect_allXorb_allLambdas_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    MPI_Comm blacscomm = pSPARC->blacscomm;
    int blacscommRank, blacscommSize;
    MPI_Comm_rank(blacscomm, &blacscommRank);
    MPI_Comm_size(blacscomm, &blacscommSize);
    int *bandNumbers = (int *)calloc(sizeof(int), pSPARC->npband);
    int *bandStartIndices = (int *)calloc(sizeof(int), pSPARC->npband);
    MPI_Allgather(&pSPARC->Nband_bandcomm, 1, MPI_INT, bandNumbers, 1, MPI_INT, blacscomm);
    MPI_Allgather(&pSPARC->band_start_indx, 1, MPI_INT, bandStartIndices, 1, MPI_INT, blacscomm);
    int *XorbLengths = (int *)calloc(sizeof(int), pSPARC->npband);
    int *XorbStartIndices = (int *)calloc(sizeof(int), pSPARC->npband);
    for (int i = 0; i < pSPARC->npband; i++) {
        XorbLengths[i] = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * bandNumbers[i];
        XorbStartIndices[i] = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * bandStartIndices[i];
    }
    // collect all Xorb
    int localXorbLength = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nband_bandcomm;
    int allXorbLength = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nstates;
    MPI_Allgatherv(&pSPARC->Xorb[0], localXorbLength, MPI_DOUBLE, &pRPA->allXorb[0], XorbLengths, XorbStartIndices, MPI_DOUBLE, blacscomm);
    if (pSPARC->Nspin_spincomm > 1) { // modify the sequence of Xorbs to collect psis with the same spin
        double *temporaryAllXorb = (double *)calloc(sizeof(double), allXorbLength);
        memcpy(temporaryAllXorb, pRPA->allXorb, sizeof(double) * allXorbLength);
        for (int band = 0; band < pSPARC->Nstates; band++) {
            memcpy(&pRPA->allXorb[band * pSPARC->Nd_d_dmcomm], &temporaryAllXorb[band * pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm], sizeof(double) * pSPARC->Nd_d_dmcomm);
            memcpy(&pRPA->allXorb[pSPARC->Nd_d_dmcomm * pSPARC->Nstates + band * pSPARC->Nd_d_dmcomm], &temporaryAllXorb[(band * pSPARC->Nspin_spincomm + 1) * pSPARC->Nd_d_dmcomm], sizeof(double) * pSPARC->Nd_d_dmcomm);
        }
        free(temporaryAllXorb);
    }

    double *allOccs = (double*)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nstates);
    pRPA->NoccupiedOrbital = 0;
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        memcpy(&(allOccs[spin * pSPARC->Nstates]), &(pSPARC->occ[spin * pSPARC->Nstates]), sizeof(double) * pSPARC->Nstates);
    }
    for (int orbital = 0; orbital < pSPARC->Nstates; orbital++) {
        if (allOccs[orbital] < 1e-4) {
            pRPA->NoccupiedOrbital = orbital;
            break;
        }
    }
    // if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (blacscommRank == blacscommSize - 1)) { // for checking the correctness
    //     char collectFileName[100];
    //     snprintf(collectFileName, 100, "nuChi0Eigscomm%d_blacsRank%d.Xorb", pRPA->nuChi0EigscommIndex, blacscommRank);
    //     FILE *outputXorb = fopen(collectFileName, "w");
    //     int testIndex = 2;
    //     for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
    //         for (int band = 0; band < pSPARC->Nstates; band++) {
    //             fprintf(outputXorb, "%d entry in spin %d band %d is %12.9f\n", testIndex, spin, band, pRPA->allXorb[spin * pSPARC->Nd_d_dmcomm * pSPARC->Nstates + band * pSPARC->Nd_d_dmcomm + testIndex]);
    //         }
    //     }
    //     fclose(outputXorb);
    // }
    free(bandNumbers);
    free(bandStartIndices);
    free(XorbLengths);
    free(XorbStartIndices);
    free(allOccs);
}

void collect_allXorb_allLambdas_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    MPI_Comm blacscomm = pSPARC->blacscomm;
    int blacscommRank, blacscommSize;
    MPI_Comm_rank(blacscomm, &blacscommRank);
    MPI_Comm_size(blacscomm, &blacscommSize);
    int *bandNumbers = (int *)calloc(sizeof(int), pSPARC->npband);
    int *bandStartIndices = (int *)calloc(sizeof(int), pSPARC->npband);
    MPI_Allgather(&pSPARC->Nband_bandcomm, 1, MPI_INT, bandNumbers, 1, MPI_INT, blacscomm);
    MPI_Allgather(&pSPARC->band_start_indx, 1, MPI_INT, bandStartIndices, 1, MPI_INT, blacscomm);
    int *XorbLengths = (int *)calloc(sizeof(int), pSPARC->npband);
    int *XorbStartIndices = (int *)calloc(sizeof(int), pSPARC->npband);
    for (int i = 0; i < pSPARC->npband; i++) {
        XorbLengths[i] = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * bandNumbers[i];
        XorbStartIndices[i] = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * bandStartIndices[i];
    }
    // collect all Xorb
    int localXorbLengthAkpt = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nband_bandcomm;
    int allXorbLengthAkpt = pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nstates;
    for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
        int localStartIndex = kpt * pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nband_bandcomm;
        int allStartIndex = kpt * pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nstates;
        MPI_Allgatherv(&pSPARC->Xorb_kpt[localStartIndex], localXorbLengthAkpt, MPI_DOUBLE_COMPLEX, &pRPA->allXorb_kpt[allStartIndex], XorbLengths, XorbStartIndices, MPI_DOUBLE_COMPLEX, blacscomm);
    }
    // MPI_Allgatherv(&pSPARC->Xorb_kpt[0], localXorbLength, MPI_DOUBLE_COMPLEX, &pRPA->allXorb_kpt[0], XorbLengths, XorbStartIndices, MPI_DOUBLE_COMPLEX, blacscomm);
    if (pSPARC->Nspin_spincomm > 1) { // modify the sequence of Xorbs to collect psis with the same spin
        int allXorbLength = allXorbLengthAkpt*pSPARC->Nkpts_kptcomm;
        double _Complex *temporaryAllXorb = (double _Complex*)calloc(sizeof(double _Complex), allXorbLength);
        memcpy(temporaryAllXorb, pRPA->allXorb_kpt, sizeof(double _Complex) * allXorbLength);
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            for (int band = 0; band < pSPARC->Nstates; band++) {
                int indexSpinUpReorder = (kpt*pSPARC->Nstates + band) * pSPARC->Nd_d_dmcomm;
                int indexSpinUp = (kpt*pSPARC->Nstates*pSPARC->Nspin_spincomm + band*pSPARC->Nspin_spincomm) * pSPARC->Nd_d_dmcomm;
                memcpy(&pRPA->allXorb_kpt[indexSpinUpReorder], &temporaryAllXorb[indexSpinUp], sizeof(double _Complex) * pSPARC->Nd_d_dmcomm);
                int indexSpinDnReorder = (pSPARC->Nkpts_kptcomm*pSPARC->Nstates + (kpt*pSPARC->Nstates + band)) * pSPARC->Nd_d_dmcomm;
                int indexSpinDn = (kpt*pSPARC->Nstates*pSPARC->Nspin_spincomm + band*pSPARC->Nspin_spincomm + 1) * pSPARC->Nd_d_dmcomm;
                memcpy(&pRPA->allXorb_kpt[indexSpinDnReorder], &temporaryAllXorb[indexSpinDn], sizeof(double _Complex) * pSPARC->Nd_d_dmcomm);
            }
        }
        free(temporaryAllXorb);
    }

    // int globalRank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    // if (globalRank == 0) {
    //     char collectedXornFileName[100];
    //     snprintf(collectedXornFileName, 100, "collected_allXorb.orbit");
    //     FILE *collectedXord = fopen(collectedXornFileName, "a");
    //     if (collectedXord ==  NULL) {
    //         printf("error printing initial delta psi\n");
    //         exit(EXIT_FAILURE);
    //     } else {
    //         for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
    //             fprintf(collectedXord, "allLambdas_localKpt%d\n", kpt);
    //             for (int psi = 0; psi < pSPARC->Nstates; psi++) {
    //                 fprintf(collectedXord, "%12.9f, ", pRPA->allLambdas[kpt*pSPARC->Nstates + psi]);
    //             }
    //             fprintf(collectedXord, "\n");
    //             fprintf(collectedXord, "allXorb_localKpt%d\n", kpt);
    //             int DMnd = pSPARC->Nd_d_dmcomm;
    //             for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
    //                 for (int psi = 0; psi < pSPARC->Nstates; psi++) {
    //                     fprintf(collectedXord, "%12.9f + %12.9fi, ", creal(pRPA->allXorb_kpt[kpt*pSPARC->Nstates*DMnd + psi*DMnd + index]), cimag(pRPA->allXorb_kpt[kpt*pSPARC->Nstates*DMnd + psi*DMnd + index]));
    //                 }
    //                 fprintf(collectedXord, "\n");
    //             }
    //         }
    //     }
    //     fclose(collectedXord);
    // }
    
    free(bandNumbers);
    free(bandStartIndices);
    free(XorbLengths);
    free(XorbStartIndices);
}

void send_recv_allXorb_allLambdas_kPq(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex) {
    MPI_Comm blacscomm = pSPARC->blacscomm;
    int blacscommRank, blacscommSize;
    MPI_Comm_rank(blacscomm, &blacscommRank);
    MPI_Comm_size(blacscomm, &blacscommSize);

    int *kptStartIndices = (int *)calloc(sizeof(int), pSPARC->npkpt + 1);
    MPI_Allgather(&pSPARC->kpt_start_indx, 1, MPI_INT, kptStartIndices, 1, MPI_INT, pSPARC->kpt_bridge_comm);
    kptStartIndices[pSPARC->npkpt] = pSPARC->Nkpts_sym + 1; // Nkpts_sym in pSPARC is the number of k-points in complete grid, no symmetry.
    // find out receiver k-point communicator
    int *kMqKptcommIndex = (int *)calloc(sizeof(int), pSPARC->Nkpts_kptcomm);
    for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
        int globalKpt = kpt + pSPARC->kpt_start_indx;
        int kMqIndex = pRPA->kMqList[globalKpt][qptIndex];
        for (int kptcommIndex = 0; kptcommIndex < pSPARC->npkpt; kptcommIndex++) {
            if ((kMqIndex >= kptStartIndices[kptcommIndex]) && (kMqIndex < kptStartIndices[kptcommIndex + 1])) {
                kMqKptcommIndex[kpt] = kptcommIndex;
            }
        }
    }
    
    // find out sender k-point communicator
    int *kPqKptcommIndex = (int *)calloc(sizeof(int), pSPARC->Nkpts_kptcomm);
    for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
        int globalKpt = kpt + pSPARC->kpt_start_indx;
        int kPqIndex = pRPA->kPqList[globalKpt][qptIndex];
        for (int kptcommIndex = 0; kptcommIndex < pSPARC->npkpt; kptcommIndex++) {
            if ((kPqIndex >= kptStartIndices[kptcommIndex]) && (kPqIndex < kptStartIndices[kptcommIndex + 1])) {
                kPqKptcommIndex[kpt] = kptcommIndex;
            }
        }
    }

    // char collectFileName[100];
    // snprintf(collectFileName, 100, "nuChi0Eigscomm%d_kptcomm%d_bandcomm%d_blacsRank%d.XorbKpt", pRPA->nuChi0EigscommIndex, pSPARC->kptcomm_index, pSPARC->bandcomm_index, blacscommRank);
    // FILE *outputXorb = fopen(collectFileName, "w");

    MPI_Comm spinComm = pSPARC->spincomm;
    int spinCommSize, spinCommRank;
    MPI_Comm_rank(spinComm, &spinCommRank);
    MPI_Comm_size(spinComm, &spinCommSize);
    MPI_Comm kptComm = pSPARC->kptcomm;
    int kptCommSize;
    MPI_Comm_size(kptComm, &kptCommSize);

    // send_recv psis
    // Irecv
    MPI_Request *recv_request = calloc(pSPARC->Nspin_spincomm*pSPARC->Nkpts_kptcomm, sizeof(MPI_Request));
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            int indexRecv = (spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates + (kpt*pSPARC->Nstates)) * pSPARC->Nd_d_dmcomm;
            int senderRank = spinCommRank + (kPqKptcommIndex[kpt] - pSPARC->kptcomm_index)*kptCommSize;
            int globalKPqIndex = (pSPARC->spin_start_indx + spin)*pSPARC->Nkpts_sym + pRPA->kPqList[pSPARC->kpt_start_indx + kpt][qptIndex];
            // fprintf(outputXorb, "Irecv local spin %d, local kpt %d paras: senderRank %d, globalKPqIndex %d\n", spin, kpt, senderRank, globalKPqIndex);
            MPI_Irecv(&pRPA->allXorb_kPq[indexRecv], pSPARC->Nstates*pSPARC->Nd_d_dmcomm, MPI_DOUBLE_COMPLEX, senderRank, globalKPqIndex, pSPARC->spincomm, &recv_request[spin*pSPARC->Nkpts_kptcomm + kpt]);
        }
    }

    // send
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            int indexSend = (spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates + (kpt*pSPARC->Nstates)) * pSPARC->Nd_d_dmcomm;
            int recverRank = spinCommRank + (kMqKptcommIndex[kpt] - pSPARC->kptcomm_index)*kptCommSize;
            int globalKptIndex = (pSPARC->spin_start_indx + spin)*pSPARC->Nkpts_sym + pSPARC->kpt_start_indx + kpt;
            // fprintf(outputXorb, "Send local spin %d, local kpt %d paras: recverRank %d, globalKptIndex %d\n", spin, kpt, recverRank, globalKptIndex);
            MPI_Send(&pRPA->allXorb_kpt[indexSend], pSPARC->Nstates*pSPARC->Nd_d_dmcomm, MPI_DOUBLE_COMPLEX, recverRank, globalKptIndex, pSPARC->spincomm);
        }
    }
    int count = 0;
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            MPI_Status status;
            MPI_Wait(&recv_request[count], &status);
            count++;
        }
    }
    free(recv_request);

    // send_recv lambdas // every processor in a kpt comm has ALL lambda and occ of the k-point!
    // Irecv
    recv_request = calloc(pSPARC->Nspin_spincomm*pSPARC->Nkpts_kptcomm, sizeof(MPI_Request));
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            int indexRecv = spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates + (kpt*pSPARC->Nstates);
            int senderRank = spinCommRank + (kPqKptcommIndex[kpt] - pSPARC->kptcomm_index)*kptCommSize;
            int globalKPqIndex = (pSPARC->spin_start_indx + spin)*pSPARC->Nkpts_sym + pRPA->kPqList[pSPARC->kpt_start_indx + kpt][qptIndex];
            // fprintf(outputXorb, "Irecv local spin %d, local kpt %d paras: senderRank %d, globalKPqIndex %d\n", spin, kpt, senderRank, globalKPqIndex);
            MPI_Irecv(&pRPA->allLambdas_kPq[indexRecv], pSPARC->Nstates, MPI_DOUBLE, senderRank, globalKPqIndex, pSPARC->spincomm, &recv_request[spin*pSPARC->Nkpts_kptcomm + kpt]);
        }
    }

    // send
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            int indexSend = spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates + (kpt*pSPARC->Nstates);
            int recverRank = spinCommRank + (kMqKptcommIndex[kpt] - pSPARC->kptcomm_index)*kptCommSize;
            int globalKptIndex = (pSPARC->spin_start_indx + spin)*pSPARC->Nkpts_sym + pSPARC->kpt_start_indx + kpt;
            // fprintf(outputXorb, "Send local spin %d, local kpt %d paras: recverRank %d, globalKptIndex %d\n", spin, kpt, recverRank, globalKptIndex);
            MPI_Send(&pSPARC->lambda[indexSend], pSPARC->Nstates, MPI_DOUBLE, recverRank, globalKptIndex, pSPARC->spincomm);
        }
    }
    
    count = 0;
    for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
            MPI_Status status;
            MPI_Wait(&recv_request[count], &status);
            count++;
        }
    }

    // int testIndex = pSPARC->Nstates - 1;
    // for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
    //     for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
    //         int globalSpin = spin + pSPARC->spin_start_indx;
    //         int globalKpt = kpt + pSPARC->kpt_start_indx;
    //         int lambdaIndex = spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates + kpt*pSPARC->Nstates + testIndex;
    //         fprintf(outputXorb, "%d lambda of spin %d, kpt %d is %12.9f\n", testIndex, globalSpin, globalKpt, pRPA->allLambdas[lambdaIndex]);
    //         for (int band = 0; band < pSPARC->Nstates; band++) {
    //             int psiIndex = spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates*pSPARC->Nd_d_dmcomm + kpt*pSPARC->Nstates*pSPARC->Nd_d_dmcomm + band * pSPARC->Nd_d_dmcomm + testIndex;
    //             fprintf(outputXorb, "%d entry in spin %d, kpt %d, band %d is %12.9f + i%12.9f\n", testIndex, globalSpin, globalKpt, band, creal(pRPA->allXorb_kpt[psiIndex]), cimag(pRPA->allXorb_kpt[psiIndex]));
    //         }
    //     }
    // }

    // for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
    //     for (int kpt = 0; kpt < pSPARC->Nkpts_kptcomm; kpt++) {
    //         int globalSpin = spin + pSPARC->spin_start_indx;
    //         int globalKpt = kpt + pSPARC->kpt_start_indx;
    //         int globalkPq = pRPA->kPqList[globalKpt][qptIndex];
    //         int lambdaIndex = spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates + kpt*pSPARC->Nstates + testIndex;
    //         fprintf(outputXorb, "%d lambda of spin %d, kPq %d (kpt %d) is %12.9f\n", testIndex, globalSpin, globalkPq, globalKpt, pRPA->allLambdas_kPq[lambdaIndex]);
    //         for (int band = 0; band < pSPARC->Nstates; band++) {
    //             int psiIndex = spin*pSPARC->Nkpts_kptcomm*pSPARC->Nstates*pSPARC->Nd_d_dmcomm + kpt*pSPARC->Nstates*pSPARC->Nd_d_dmcomm + band * pSPARC->Nd_d_dmcomm + testIndex;
    //             fprintf(outputXorb, "%d entry in spin %d, kPq %d (kpt %d), band %d is %12.9f + i%12.9f\n", testIndex, globalSpin, globalkPq, globalKpt, band, creal(pRPA->allXorb_kPq[psiIndex]), cimag(pRPA->allXorb_kPq[psiIndex]));
    //         }
    //     }
    // }

    // fclose(outputXorb);

    free(kptStartIndices);
    free(kPqKptcommIndex);
    free(kMqKptcommIndex);
    free(recv_request);
}