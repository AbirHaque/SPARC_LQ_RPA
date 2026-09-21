#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <assert.h>
// this is for checking existence of files

#include "prepare_PQ_operators.h"
#include "parallelization_RPA.h"
#include "parallelization.h"

void prepare_PQ_operators(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (flagNoDmcomm) return;
    if (pRPA->nuChi0EigscommIndex < 0)
        return;
    int rank;
    int Ns = pSPARC->Nstates;
    MPI_Comm blacscomm = pSPARC->blacscomm;
    MPI_Comm_rank(blacscomm, &rank);
    if (pSPARC->isGammaPoint) {
        int *bandNumbers = (int*)calloc(sizeof(int), pSPARC->npband);
        int *bandStartIndices = (int*)calloc(sizeof(int), pSPARC->npband);
        MPI_Allgather(&pSPARC->Nband_bandcomm, 1, MPI_INT, bandNumbers, 1, MPI_INT, blacscomm);
        MPI_Allgather(&pSPARC->band_start_indx, 1, MPI_INT, bandStartIndices, 1, MPI_INT, blacscomm);
        for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
            MPI_Allgatherv(&pSPARC->lambda[spin*pSPARC->Nband_bandcomm], pSPARC->Nband_bandcomm, MPI_DOUBLE, pRPA->allEpsilonsGamma, bandNumbers, bandStartIndices, MPI_DOUBLE, blacscomm);
        }

        int *neighborBandStartEnd = pRPA->neighborBandStartEnd;  // currently, we don't treat spin-up and spin-dn seperately; nearby bands judged by only spin-up or no spin eigenvalues.
        neighborBandStartEnd[0] = 0; neighborBandStartEnd[1] = 0;
        int *allBandComms = (int*)calloc(sizeof(int), Ns);
        int amountNeighborBands = find_nearby_band_indices_gamma(pSPARC, bandStartIndices, pRPA->allEpsilonsGamma, pRPA->nearbyBandIndicesGamma, neighborBandStartEnd, allBandComms);
        printf("nuChi0Eigscomm %d, bandcomm %d, amountNeighborBands %d, neighborBandStartEnd[0], [1] %d, %d\n", pRPA->nuChi0EigscommIndex,
            pSPARC->bandcomm_index, amountNeighborBands, neighborBandStartEnd[0], neighborBandStartEnd[1]);
        if ((!pRPA->nuChi0EigscommIndex) && (!rank)) {
            printf("I am rank %d in nuChi0Eigscomm %d, the last 5 bands belong to comm %d %d %d %d %d\n", rank, pRPA->nuChi0EigscommIndex,
                allBandComms[Ns - 5], allBandComms[Ns - 4], allBandComms[Ns - 3], allBandComms[Ns - 2], allBandComms[Ns - 1]);
        }

        pRPA->neighborBandIndicesGamma = (int*)calloc(sizeof(int), amountNeighborBands);
        pRPA->neighborBandsGamma = (double*)calloc(sizeof(double), pSPARC->Nspin_spincomm*pSPARC->Nd_d_dmcomm*amountNeighborBands);
        get_neighborBands_gamma(pSPARC, pRPA->nearbyBandIndicesGamma, neighborBandStartEnd, allBandComms, pRPA->neighborBandIndicesGamma, pRPA->neighborBandsGamma);

        free(bandNumbers);
        free(bandStartIndices);
        free(allBandComms);
    } else {

    }
}

int find_nearby_band_indices_gamma(SPARC_OBJ *pSPARC, int *bandStartIndices, double *allEpsilonsGamma, int *nearbyBandIndices, int *neighborBandStartEnd, int *allBandComms) { // find nearby bands of every band in the band_comm, or this processor
    double *allEpsilons = allEpsilonsGamma;
    int Ns = pSPARC->Nstates;
    int ncol = pSPARC->Nband_bandcomm;
    
    for (int comm = 0; comm < pSPARC->npband; comm++) {
        int theCommStartIndex = bandStartIndices[comm];
        for (int band = theCommStartIndex; band < Ns; band++) {
            allBandComms[band] = comm;
        }
    }
    int amountNeighborBands = 0;
    // for (int spin = 0; spin < pSPARC->Nspin_spincomm; spin++) {
        int spin = 0; // currently, we don't treat spin-up and spin-dn seperately; nearby bands judged by only spin-up or no spin eigenvalues.
        for (int band = 0; band < ncol; band++) {
            double theEpsilon = pSPARC->lambda[spin*ncol + band];
            int flagLeft = 0;
            for (int band2 = 0; band2 < Ns; band2++) {
                double band2Ep = allEpsilons[band2];
                if (fabs(theEpsilon - band2Ep) < 6e-2) {
                    if (!flagLeft) {
                        nearbyBandIndices[spin*2*ncol + 2*band] = band2;
                        flagLeft = 1;
                    }
                    nearbyBandIndices[spin*2*ncol + 2*band + 1] = band2;
                }
            }
        }
        int minNearbyIndex = Ns;
        int maxNearbyIndex = 0;
        for (int band = 0; band < ncol; band++) {
            if (nearbyBandIndices[spin*2*ncol + 2*band] < minNearbyIndex) minNearbyIndex = nearbyBandIndices[spin*2*ncol + 2*band];
            if (nearbyBandIndices[spin*2*ncol + 2*band + 1] > maxNearbyIndex) maxNearbyIndex = nearbyBandIndices[spin*2*ncol + 2*band + 1];
        }
        neighborBandStartEnd[spin*2] = minNearbyIndex;
        neighborBandStartEnd[spin*2 + 1] = maxNearbyIndex;
        amountNeighborBands += maxNearbyIndex - minNearbyIndex - ncol + 1;
    // }
    return amountNeighborBands;
}

void get_neighborBands_gamma(SPARC_OBJ *pSPARC, int *nearbyBandIndices, int *neighborBandStartEnd, int *allBandComms, 
    int *neighborBandIndices, double *neighborBands) {
    MPI_Comm blacscomm = pSPARC->blacscomm;
    int rank;
    MPI_Comm_rank(pSPARC->blacscomm, &rank);
    int Ns = pSPARC->Nstates;
    int Nspin = pSPARC->Nspin_spincomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nd_d = pSPARC->Nd_d_dmcomm;
    MPI_Request *send_request = NULL, *recv_request = NULL;
    int *bandStartIndices = (int*)calloc(sizeof(int), pSPARC->npband);
    MPI_Allgather(&pSPARC->band_start_indx, 1, MPI_INT, bandStartIndices, 1, MPI_INT, blacscomm);
    
    int neighborBandStart = neighborBandStartEnd[0];
    int neighborBandEnd = neighborBandStartEnd[1];
    int commStart = allBandComms[neighborBandStart];
    int commEnd = allBandComms[neighborBandEnd];

    if (commStart == commEnd) {
        return;
    }
    
    // send buffs
    send_request = malloc((commEnd - commStart + 1) * sizeof(MPI_Request));
    for (int recvComm = commStart; recvComm < commEnd + 1; recvComm++) {
        int amountBandsToSend = 0;
        if (recvComm < rank) {
            int recvCommBandEnd = (recvComm == pSPARC->npband - 1) ? (Ns - 1) : (bandStartIndices[recvComm + 1] - 1);
            for (int theBand = 0; theBand < ncol; theBand++) {
                int theBandNearbyStart = nearbyBandIndices[2*theBand];
                if (theBandNearbyStart <= recvCommBandEnd) {
                    amountBandsToSend++;
                }
            }
            // printf("rank %d, send to %d %d bands\n", rank, recvComm, amountBandsToSend);
            if (amountBandsToSend) {
                MPI_Isend(pSPARC->Xorb, Nspin*amountBandsToSend*Nd_d, MPI_DOUBLE, recvComm, rank, blacscomm, &(send_request[recvComm - commStart]));
            }
        }
        else if (recvComm == rank) continue;
        else {
            int recvCommBandStart = bandStartIndices[recvComm];
            for (int theBand = 0; theBand < ncol; theBand++) {
                int theBandNearbyEnd = nearbyBandIndices[2*theBand + 1];
                if (theBandNearbyEnd >= recvCommBandStart) {
                    amountBandsToSend++;
                }
            }
            // printf("rank %d, send to %d %d bands\n", rank, recvComm, amountBandsToSend);
            if (amountBandsToSend) {
                MPI_Isend(pSPARC->Xorb + Nspin*(ncol - amountBandsToSend)*Nd_d, Nspin*amountBandsToSend*Nd_d, MPI_DOUBLE, recvComm, rank, blacscomm, &(send_request[recvComm - commStart]));
            }
        }
    }
    // recv buffs
    recv_request = malloc((commEnd - commStart + 1) * sizeof(MPI_Request));
    int despBandsToRecv = 0; int amountBandsToRecv = 0;
    for (int sendComm = commStart; sendComm < commEnd + 1; sendComm++) {
        int sendCommBandStart = bandStartIndices[sendComm];
        int sendCommBandEnd = (sendComm == pSPARC->npband - 1) ? (Ns - 1) : (bandStartIndices[sendComm + 1] - 1);
        if (sendComm < rank) {
            amountBandsToRecv = (sendCommBandEnd + 1 - neighborBandStart) > 0 ? (sendCommBandEnd + 1 - neighborBandStart) : 0;
            amountBandsToRecv -= (sendCommBandStart - neighborBandStart) > 0 ? (sendCommBandStart - neighborBandStart) : 0;
        }
        else if (sendComm == rank) continue;
        else {
            amountBandsToRecv = (neighborBandEnd - sendCommBandStart + 1) > 0 ? (neighborBandEnd - sendCommBandStart + 1) : 0;
            amountBandsToRecv -= (neighborBandEnd - sendCommBandEnd) > 0 ? (neighborBandEnd - sendCommBandEnd) : 0;
        }
        // printf("rank %d, recv from %d %d bands to disp %d\n", rank, sendComm, amountBandsToRecv, despBandsToRecv);
        if (amountBandsToRecv) {
            MPI_Irecv(neighborBands + despBandsToRecv*Nspin*Nd_d, Nspin*amountBandsToRecv*Nd_d, MPI_DOUBLE, sendComm, sendComm, blacscomm, &(recv_request[sendComm - commStart]));
        }
        despBandsToRecv += amountBandsToRecv;
    }

    int pos = 0;
    int myBandStartIndex = bandStartIndices[rank];
    int myBandEndIndex = (rank == pSPARC->npband - 1) ? (Ns - 1) : (bandStartIndices[rank + 1] - 1);
    for (int band = neighborBandStart; band < myBandStartIndex; band++) {
        neighborBandIndices[pos] = band;
        pos++;
    }
    for (int band = myBandEndIndex + 1; band < neighborBandEnd + 1; band++) {
        neighborBandIndices[pos] = band;
        pos++;
    }
    free(bandStartIndices);
    free(send_request);
    free(recv_request);
}