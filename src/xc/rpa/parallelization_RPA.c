/**
 * @file    parallelization_RPA.c
 * @brief   This file contains functions for setting up nuChi0 comms. Every nuChi0 comm handles their assigned trial vectors.
 *          nuChi0 comms hold the top level of parallelization framework.
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
#include <limits.h>

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

#include <initialization.h>
#include "eigenSolver.h"
#include "tools.h"

#include "parallelization_RPA.h"
#include "parallelization.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

void Setup_Comms_RPA(RPA_OBJ *pRPA, int Nspin, int Nkpts, int Nstates, int Nd) {
    int nprocWorld, rankWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    // The Sternheimer equation will be solved in pSPARC. pRPA is the structure saving variables not in pSPARC.
    dims_divide_Eigs(nprocWorld, rankWorld, pRPA->nuChi0Neig, &pRPA->npnuChi0Neig);
    // 3. nuChi0Eigs communicator, distribute all trial eigenvectors of nuChi0, saved in pRPA
    pRPA->npnuChi0Neig = judge_npObject(pRPA->nuChi0Neig, nprocWorld, pRPA->npnuChi0Neig);
    pRPA->nNuChi0Eigscomm = distribute_comm_load_LQ(pRPA->nuChi0Neig, pRPA->npnuChi0Neig, rankWorld, nprocWorld, &(pRPA->nuChi0EigscommIndex), &(pRPA->nuChi0EigsStartIndex), &(pRPA->nuChi0EigsEndIndex));//HAQUE TODO: Have a check to see if we are using LQ or SQ. If LQ, use distribute_comm_load_LQ, else distribute_comm_load
    int maxBlockSize = pRPA->SternBlockSize[0];
    int minBlockSize = pRPA->SternBlockSize[0];
    for (int i = 1; i < pRPA->Nomega; i++) {
        if (pRPA->SternBlockSize[i] > maxBlockSize) {
            maxBlockSize = pRPA->SternBlockSize[i];
        } 
        if (pRPA->SternBlockSize[i] < minBlockSize) {
            minBlockSize = pRPA->SternBlockSize[i];
        }
    }
    if (!rankWorld) {
        printf("global rank %d, maxBlockSize %d, minBlockSize %d\n", rankWorld, maxBlockSize, minBlockSize);
    }
    if ((minBlockSize < 0) || (maxBlockSize > pRPA->nNuChi0Eigscomm)) {
        if (!rankWorld) {
            printf("WARNING: the input block sizes are not valid (either not input, or input not in range [1, pRPA->nNuChi0Eigscomm]).\n"
                " Now they are replaced by pRPA->nNuChi0Eigscomm.\n");
        }
        int defaultBlockSize = (pRPA->nNuChi0Eigscomm < (Nd/12)) ? pRPA->nNuChi0Eigscomm : (Nd/12 + 1);
        for (int i = 0; i < pRPA->Nomega; i++) {
            pRPA->SternBlockSize[i] = defaultBlockSize;
        }
    }
    int color = (pRPA->nuChi0EigscommIndex >= 0) ? pRPA->nuChi0EigscommIndex : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &pRPA->nuChi0Eigscomm);

    pRPA->rank0nuChi0EigscommInWorld = rankWorld;
    MPI_Bcast(&pRPA->rank0nuChi0EigscommInWorld, 1, MPI_INT, 0, pRPA->nuChi0Eigscomm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d nuChi0Eigscomm, I will handle nuChi0Eigs %d ~ %d\n", rankWorld, pRPA->nuChi0EigscommIndex, pRPA->nuChi0EigsStartIndex, pRPA->nuChi0EigsEndIndex);
    #endif
    // 3.3 nuChi0Eigs bridge communicator, connect all processors in different nuChi0Eigs communicator having the same index, to broadcast eigenvalues and eigenvectors from DFT
    int nprocNuChi0EigsComm, rankNuChi0EigsComm;
    MPI_Comm_size(pRPA->nuChi0Eigscomm, &nprocNuChi0EigsComm);
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rankNuChi0EigsComm);
    int judgeJoinCompute = (pRPA->nuChi0EigscommIndex >= 0); // if it is 0, the processor will not join computation
    color = (judgeJoinCompute > 0) ? rankNuChi0EigsComm : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, pRPA->nuChi0EigscommIndex, &pRPA->nuChi0EigsBridgeComm);
    pRPA->nuChi0EigsBridgeCommIndex = (judgeJoinCompute > 0) ? rankNuChi0EigsComm : -1;
    int rankNuChi0EigsBridgeComm; 
    MPI_Comm_rank(pRPA->nuChi0EigsBridgeComm, &rankNuChi0EigsBridgeComm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d nuChi0EigsBridgeComm, My rank in it is %d\n", rankWorld, pRPA->nuChi0EigsBridgeCommIndex, rankNuChi0EigsBridgeComm);
    #endif
    
    // 4. every nuChi0Eigs communicator replace MPI_COMM_WORLD in Setup_Comms function of SPARC, call Setup_Comms_SPARC in RPA
    // 5. spin communicator, in pSPARC
    // 6. k-point communicator, use ALL k-points, no symmetric reduction, saved in SPARC
    // 7. band communicator, in pSPARC
    // 8. domain communicator, in pSPARC
}

void dims_divide_Eigs(int nprocWorld, int rankWorld, int nuChi0Neig, int *npnuChi0Neig) {
    // this function is for computing the optimal dividance on nuChi0Eigs, (spins, kpts and bands for pSPARC).
    if ((*npnuChi0Neig > nprocWorld) || (*npnuChi0Neig > nuChi0Neig)) { // confined by total number of processors!
        if (!rankWorld) printf("Input NP_NUCHI_EIGS_PARAL_RPA is larger than the total number of processors or number of eigenvalues nu chi0 operator to be solved,\n so it is not valid. This input is abandoned.\n");
        *npnuChi0Neig = -1;
    }
    if (*npnuChi0Neig <= 0) { // if there is valid input npnuChi0Neig, just apply it
        *npnuChi0Neig = nprocWorld < nuChi0Neig ? nprocWorld : nuChi0Neig; // distribute nuChi0Neig first
    }
}

int judge_npObject(int nObjectInTotal, int sizeFatherComm, int npInput) {
    int npOutput = npInput;
    int maxLimit = min(sizeFatherComm, nObjectInTotal);
    if (npOutput == -1) {
        npOutput = maxLimit;
    } else if (npOutput > maxLimit) {
        npOutput = maxLimit;
    }

    int avgNobjectAfterP = nObjectInTotal / npOutput;
    if (nObjectInTotal % npOutput) avgNobjectAfterP++;
    // printf("npIutput %d, maxLimit %d, decided npOutput %d\n", npInput, maxLimit, npOutput);
    return npOutput;
}

int distribute_comm_load(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex) {
    int sizeComm = sizeFatherComm / npObject;
    if (rankFatherComm < (sizeFatherComm - sizeFatherComm % npObject))
        *commIndex = rankFatherComm / sizeComm;
    else
        *commIndex = -1;
    
    int nEig, nObjectInComm; // for block cyclic
    nEig = (nObjectInTotal - 1) / npObject + 1; // this is equal to ceil(Nstates/npband), for int inputs only
    nObjectInComm = *commIndex < (nObjectInTotal / nEig) ? nEig : (*commIndex == (nObjectInTotal / nEig) ? (nObjectInTotal % nEig) : 0);

    if (*commIndex == -1) {
        *objectStartIndex = 0;
    } else if (*commIndex <= (nObjectInTotal / nEig)) {
        *objectStartIndex = *commIndex * nEig;
    } else {
        *objectStartIndex = nObjectInTotal;
    }
    *objectEndIndex = *objectStartIndex + nObjectInComm - 1;
    return nObjectInComm;
}
int distribute_comm_load_LQ(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex) {
    int sizeComm = sizeFatherComm / npObject;
    if (rankFatherComm < (npObject * sizeComm)) {
        *commIndex = rankFatherComm / sizeComm;
    } else {
        *commIndex = -1;
    }
    if (*commIndex == -1 || nObjectInTotal <= 0) {
        *objectStartIndex = 0;
        *objectEndIndex = -1;
        return 0;
    }
    int base = nObjectInTotal / npObject;
    int remainder = nObjectInTotal % npObject;
    if (*commIndex < remainder) {
        *objectStartIndex = *commIndex * (base + 1);
        *objectEndIndex = *objectStartIndex + (base + 1) - 1;
    } else {
        *objectStartIndex = *commIndex * base + remainder;
        *objectEndIndex = *objectStartIndex + base - 1;
    }
    int nObjectInComm = (*objectEndIndex - *objectStartIndex + 1);
    if (nObjectInComm < 0) nObjectInComm = 0;

    return nObjectInComm;
}

void setup_blacsComm_RPA(RPA_OBJ *pRPA, int flagNoDmcomm, int DMnd, int Nspinor_spincomm, int Nspinor_eig, 
    int Nd, int MAX_NS, int eig_paral_blksz, int isGammaPoint, int printFlag) {
    int nprocWorld, rankWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    int nprocNuChi0EigsComm = -1;
    int color;
    int dims[3] = {0, 0, 0};
    if (pRPA->nuChi0EigscommIndex < 0 || flagNoDmcomm) {
        color = INT_MAX;
    } else {
        MPI_Comm_size(pRPA->nuChi0Eigscomm, &nprocNuChi0EigsComm);
        int rank_nuChi0Eigscomm;
        MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank_nuChi0Eigscomm);
        color = rank_nuChi0Eigscomm;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, pRPA->nuChi0EigscommIndex, &pRPA->nuChi0BlacsComm);

    // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int size_blacscomm, DMndsp, DMndspe;
    int *usermap, *usermap_0, *usermap_1;
    int info, bandsizes[2], nprow, npcol, myrow, mycol;

    size_blacscomm = pRPA->npnuChi0Neig;
    DMndsp = DMnd * Nspinor_spincomm;
    DMndspe = DMnd * Nspinor_eig;
    FILE *outputFile = NULL;
    #ifdef DEBUG
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm) && (printFlag)) {
        outputFile = fopen(pRPA->timeRecordname, "a");
    }
    #endif
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        usermap = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_0 = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_1 = (int *)malloc(sizeof(int)*size_blacscomm);
        for (int i = 0; i < size_blacscomm; i++) {
            usermap[i] = usermap_0[i] = usermap_1[i] = rankWorld - pRPA->rank0nuChi0EigscommInWorld + i*nprocNuChi0EigsComm;
        }

        // in order to use a subgroup of blacscomm, use the following
        // to get a good number of subgroup processes
        bandsizes[0] = Nd * Nspinor_eig;
        bandsizes[1] = pRPA->nuChi0Neig;
        ScaLAPACK_Dims_2D_BLCYC(size_blacscomm, bandsizes, dims);
        #ifdef DEBUG
        if (printFlag) fprintf(outputFile, "global rank = %d, size_blacscomm = %d, nuChi0 ScaLAPACK topology %d, Dims = (%d, %d)\n", rankWorld, size_blacscomm, color, dims[0], dims[1]);
        #endif
        // TODO: make it able to use a subgroup of the blacscomm! For now just enforce it.
        if (dims[0] * dims[1] != size_blacscomm) {
            dims[0] = size_blacscomm;
            dims[1] = 1;
        }
    } else {
        usermap = (int *)malloc(sizeof(int)*1);
        usermap_0 = (int *)malloc(sizeof(int)*1);
        usermap_1 = (int *)malloc(sizeof(int)*1);
        usermap[0] = usermap_0[0] = usermap_1[0] = rankWorld;
        dims[0] = dims[1] = 1;
    }

    int myrank_mpi, nprocs_mpi;
    Cblacs_pinfo( &myrank_mpi, &nprocs_mpi );

    // the following commands will create a context with handle ictxt_blacs
    Cblacs_get( -1, 0, &pRPA->ictxt_blacs );
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        Cblacs_gridmap( &pRPA->ictxt_blacs, usermap_0, 1, 1, pRPA->npnuChi0Neig); // row topology
    } else {
        Cblacs_gridmap( &pRPA->ictxt_blacs, usermap_0, 1, 1, dims[0] * dims[1]); // row topology
    }
    free(usermap_0);


    // get coord of each process in original context
    Cblacs_gridinfo( pRPA->ictxt_blacs, &nprow, &npcol, &myrow, &mycol );

    int ZERO = 0, mb, nb, llda;
    mb = max(1, DMndspe);
    nb = (pRPA->nuChi0Neig - 1) / pRPA->npnuChi0Neig + 1; // equal to ceil(Nstates/npband), for int only
    // set up descriptor for storage of orbitals in ictxt_blacs (original)
    llda = max(1, DMndsp);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        descinit_(&pRPA->desc_orbitals[0], &DMndspe, &pRPA->nuChi0Neig,
                  &mb, &nb, &ZERO, &ZERO, &pRPA->ictxt_blacs, &llda, &info);
    } else {
        for (int i = 0; i < 9; i++)
            pRPA->desc_orbitals[i] = 0;
    }
    #ifdef DEBUG
    int temp_r, temp_c;
    temp_r = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
    temp_c = numroc_( &pRPA->nuChi0Neig, &nb, &mycol, &ZERO, &npcol);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm) && (printFlag))
        fprintf(outputFile, "global rank = %2d, 1D topo, my nuChi0 blacs rank = %d, BLCYC size (%d, %d), actual size (%d, %d), DMndspe %d, mb %d, myrow %d, nprow %d, nuChi0Neig %d, nb %d, mycol %d, npcol %d\n", 
            rankWorld, pRPA->nuChi0EigscommIndex, temp_r, temp_c, DMndsp, pRPA->nNuChi0Eigscomm,
            DMndspe, mb, myrow, nprow, pRPA->nuChi0Neig, nb, mycol, npcol);
    #endif

    // compose descripter for permutation matrix in 1D topo
    int mbP, nbP, lldaP;
    mbP = max(1, pRPA->nuChi0Neig);
    nbP = (pRPA->nuChi0Neig - 1) / pRPA->npnuChi0Neig + 1;
    lldaP = max(1, pRPA->nuChi0Neig);
    pRPA->rPermute = 0; pRPA->cPermute = 0;
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        descinit_(&pRPA->desc_permute[0], &pRPA->nuChi0Neig, &pRPA->nuChi0Neig,
                  &mbP, &nbP, &ZERO, &ZERO, &pRPA->ictxt_blacs, &lldaP, &info);
        // printf("parallelization_RPA.c, line 253, desc_permute %p %p %p %p %p \n", &pRPA->nuChi0Neig, &mbP, &myrow, &ZERO, &nprow);
        pRPA->rPermute = numroc_( &pRPA->nuChi0Neig, &mbP, &myrow, &ZERO, &nprow);
        pRPA->cPermute = numroc_( &pRPA->nuChi0Neig, &nbP, &mycol, &ZERO, &npcol);
        // printf("parallelization_RPA.c, line 253, mbP %d nbP %d nprow %d npcol %d, rPermute %d, cPermute %d \n", mbP, nbP, nprow, npcol, pRPA->rPermute, pRPA->cPermute);
    } else {
        for (int i = 0; i < 9; i++)
            pRPA->desc_permute[i] = 0;
    }
    if (isGammaPoint) {
        int rP = max(pRPA->rPermute, 1);
        int cP = max(pRPA->cPermute, 1);
        pRPA->permute = (double*)calloc(sizeof(double), rP*cP);
        assert(pRPA->permute);
    } else {
        int rP = max(pRPA->rPermute, 1);
        int cP = max(pRPA->cPermute, 1);
        pRPA->permute_kpt = (double _Complex *)calloc(sizeof(double _Complex), rP*cP);
        assert(pRPA->permute_kpt);
    }
    #ifdef DEBUG
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm) && (printFlag))
        fprintf(outputFile, "global rank = %2d, 1D topo, my Permute blacs rank = %d, BLCYC size (%d, %d), nuChi0Neig %d, mbP %d, myrow %d, nprow %d, nuChi0Neig %d, nbP %d, mycol %d, npcol %d\n", 
            rankWorld, pRPA->nuChi0EigscommIndex, pRPA->rPermute, pRPA->cPermute,
            pRPA->nuChi0Neig, mbP, myrow, nprow, pRPA->nuChi0Neig, nbP, mycol, npcol);
    #endif

    // create ictxt_blacs topology
    Cblacs_get( -1, 0, &pRPA->ictxt_blacs_topo );
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        // create usermap_1 = reshape(usermap,[dims[0], dims[1]])
        for (int j = 0; j < dims[1]; j++) {
            for (int i = 0; i < dims[0]; i++) {
                usermap_1[j*dims[0]+i] = usermap[i*dims[1]+j];
            }
        }
    }
    Cblacs_gridmap( &pRPA->ictxt_blacs_topo, usermap_1, dims[0], dims[0], dims[1] ); // Cart topology
    free(usermap_1);
    free(usermap);

    // get coord of each process in block cyclic topology context
    Cblacs_gridinfo( pRPA->ictxt_blacs_topo, &nprow, &npcol, &myrow, &mycol );
    pRPA->nprow_ictxt_blacs_topo = nprow;
    pRPA->npcol_ictxt_blacs_topo = npcol;

    // set up descriptor for block-cyclic format storage of orbitals in ictxt_blacs
    // TODO: make block-cyclic parameters mb and nb input variables!
    mb = max(1, DMndspe / dims[0]); // this is only block, no cyclic! Tune this to improve efficiency!
    nb = max(1, pRPA->nuChi0Neig / dims[1]); // this is only block, no cyclic!

    // find number of rows/cols of the local distributed orbitals
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        pRPA->nr_orb_BLCYC = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
        pRPA->nc_orb_BLCYC = numroc_( &pRPA->nuChi0Neig, &nb, &mycol, &ZERO, &npcol);
    } else {
        pRPA->nr_orb_BLCYC = 1;
        pRPA->nc_orb_BLCYC = 1;
    }
    llda = max(1, pRPA->nr_orb_BLCYC);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        descinit_(&pRPA->desc_orb_BLCYC[0], &DMndspe, &pRPA->nuChi0Neig,
                  &mb, &nb, &ZERO, &ZERO, &pRPA->ictxt_blacs_topo, &llda, &info);
    } else {
        for (int i = 0; i < 9; i++)
            pRPA->desc_orb_BLCYC[i] = 0;
    }
    #ifdef DEBUG
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm) && (printFlag))
            fprintf(outputFile, "global rank = %2d, 2D topo, my nuChi0 blacs rank = %d, BLCYC size (%d, %d), actual size (%d, %d), DMndspe %d, mb %d, myrow %d, nprow %d, nuChi0Neig %d, nb %d, mycol %d, npcol %d\n", 
                rankWorld, pRPA->nuChi0EigscommIndex, pRPA->nr_orb_BLCYC, pRPA->nc_orb_BLCYC, DMndsp, pRPA->nNuChi0Eigscomm,
                DMndspe, mb, myrow, nprow, pRPA->nuChi0Neig, nb, mycol, npcol);
    #endif
    // set up distribution of projected Hamiltonian and the corresponding overlap matrix
    // TODO: Find optimal distribution of the projected Hamiltonian and mass matrix!
    //       For now Hp and Mp are distributed as follows: we distribute them in the same
    //       context topology as the bands.
    //       Note that mb = nb!

    // the maximum Nstates up to which we will use LAPACK to solve
    // the subspace eigenproblem in serial
    // int MAX_NS = 2000;
    pRPA->eig_useLAPACK = (pRPA->nuChi0Neig <= MAX_NS) ? 1 : 0; // just for test, 20

    int mbQ, nbQ, lldaQ;
       // block size for storing Hp and Mp
    if (pRPA->eig_useLAPACK == 1) {
        // in this case we will call LAPACK instead to solve the subspace eigenproblem
        mb = nb = pRPA->nuChi0Neig;
        mbQ = nbQ = 64; // block size for storing subspace eigenvectors
    } else {
        // in this case we will use ScaLAPACK to solve the subspace eigenproblem
        mb = nb = eig_paral_blksz;
        mbQ = nbQ = eig_paral_blksz; // block size for storing subspace eigenvectors
    }

    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        pRPA->nr_Hp_BLCYC = pRPA->nr_Mp_BLCYC = numroc_( &pRPA->nuChi0Neig, &mb, &myrow, &ZERO, &nprow);
        pRPA->nr_Hp_BLCYC = pRPA->nr_Mp_BLCYC = max(1, pRPA->nr_Mp_BLCYC);
        pRPA->nc_Hp_BLCYC = pRPA->nc_Mp_BLCYC = numroc_( &pRPA->nuChi0Neig, &nb, &mycol, &ZERO, &npcol);
        pRPA->nc_Hp_BLCYC = pRPA->nc_Mp_BLCYC = max(1, pRPA->nc_Mp_BLCYC);
        pRPA->nr_Q_BLCYC = numroc_( &pRPA->nuChi0Neig, &mbQ, &myrow, &ZERO, &nprow);
        pRPA->nc_Q_BLCYC = numroc_( &pRPA->nuChi0Neig, &nbQ, &mycol, &ZERO, &npcol);
    } else {
        pRPA->nr_Hp_BLCYC = pRPA->nc_Hp_BLCYC = 1;
        pRPA->nr_Mp_BLCYC = pRPA->nc_Mp_BLCYC = 1;
        pRPA->nr_Q_BLCYC  = pRPA->nc_Q_BLCYC  = 1;
    }

    llda = max(1, pRPA->nr_Hp_BLCYC);
    lldaQ= max(1, pRPA->nr_Q_BLCYC);
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm)) {
        descinit_(&pRPA->desc_Hp_BLCYC[0], &pRPA->nuChi0Neig, &pRPA->nuChi0Neig,
                  &mb, &nb, &ZERO, &ZERO, &pRPA->ictxt_blacs_topo, &llda, &info);
        for (int i = 0; i < 9; i++) {
            pRPA->desc_Mp_BLCYC[i] = pRPA->desc_Hp_BLCYC[i];
        }
        descinit_(&pRPA->desc_Q_BLCYC[0], &pRPA->nuChi0Neig, &pRPA->nuChi0Neig,
                  &mbQ, &nbQ, &ZERO, &ZERO, &pRPA->ictxt_blacs_topo, &lldaQ, &info);
        for (int i = 0; i < 9; i++) { // permutation matrix in 2D topo is the same as Q matrix
            pRPA->desc_permute_BLCYC[i] = pRPA->desc_Q_BLCYC[i];
        }
    } else {
        for (int i = 0; i < 9; i++) {
            pRPA->desc_permute_BLCYC[i] = pRPA->desc_Q_BLCYC[i] = pRPA->desc_Mp_BLCYC[i] = pRPA->desc_Hp_BLCYC[i] = 0;
        }
    }

    #ifdef DEBUG
        if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm) && (printFlag))
            fprintf(outputFile, "global rank = %d, Hp topo, mb = %d, mbQ = %d, myrow %d, nprow %d, mycol %d, npcol %d, nr_Hp = %d, nc_Hp = %d\n", rankWorld, mb, mbQ, myrow, nprow, mycol, npcol, pRPA->nr_Hp_BLCYC, pRPA->nc_Hp_BLCYC);
    #endif

    // allocate memory for block cyclic distribution of projected Hamiltonian and mass matrix
    if (isGammaPoint){
        pRPA->Hp = (double *)malloc(pRPA->nr_Hp_BLCYC * pRPA->nc_Hp_BLCYC * sizeof(double));
        assert(pRPA->Hp);
        pRPA->Mp = (double *)malloc(pRPA->nr_Mp_BLCYC * pRPA->nc_Mp_BLCYC * sizeof(double));
        assert(pRPA->Mp);
        pRPA->Q  = (double *)malloc(pRPA->nr_Q_BLCYC * pRPA->nc_Q_BLCYC * sizeof(double));
        assert(pRPA->Q);
        pRPA->permute_BLCYC = (double *)malloc(sizeof(double) * pRPA->nr_Q_BLCYC * pRPA->nc_Q_BLCYC);
        assert(pRPA->permute_BLCYC);
    } else{
        pRPA->Hp_kpt = (double _Complex *) malloc(pRPA->nr_Hp_BLCYC * pRPA->nc_Hp_BLCYC * sizeof(double _Complex));
        assert(pRPA->Hp_kpt);
        pRPA->Mp_kpt = (double _Complex *) malloc(pRPA->nr_Mp_BLCYC * pRPA->nc_Mp_BLCYC * sizeof(double _Complex));
        assert(pRPA->Mp_kpt);
        pRPA->Q_kpt  = (double _Complex *) malloc(pRPA->nr_Q_BLCYC * pRPA->nc_Q_BLCYC * sizeof(double _Complex));
        assert(pRPA->Q_kpt);
        pRPA->permute_BLCYC_kpt = (double _Complex *)malloc(sizeof(double _Complex) * pRPA->nr_Q_BLCYC * pRPA->nc_Q_BLCYC);
        assert(pRPA->permute_BLCYC_kpt);
    }

    if (pRPA->eig_useLAPACK == 0) {
        if (pRPA->eig_paral_maxnp < 0) {
            char RorC, SorG;
            RorC = (isGammaPoint) ? 'R' : 'C';
            SorG = 'G';
            pRPA->eig_paral_maxnp = parallel_eigensolver_max_processor(pRPA->nuChi0Neig, RorC, SorG); // just for test
            // pRPA->eig_paral_maxnp = size_blacscomm;
        }
            
        int gridsizes[2] = {pRPA->nuChi0Neig, pRPA->nuChi0Neig}, ierr = 1;
        SPARC_Dims_create(min(size_blacscomm, pRPA->eig_paral_maxnp), 2, gridsizes, 1, pRPA->eig_paral_subdims, &ierr);
        if (ierr) {
            pRPA->eig_paral_subdims[0] = pRPA->eig_paral_subdims[1] = 1;
        } else { // select a number which is square of integer
            int totalNumProcessors = pRPA->eig_paral_subdims[0] * pRPA->eig_paral_subdims[1];
            int trialNum = floor(sqrt((double) totalNumProcessors));
            if ((trialNum + 1) * (trialNum + 1) < size_blacscomm) {
                pRPA->eig_paral_subdims[0] = pRPA->eig_paral_subdims[1] = trialNum + 1;
            } else {
                pRPA->eig_paral_subdims[0] = pRPA->eig_paral_subdims[1] = trialNum;
            }
        }
    #ifdef DEBUG
    if ((pRPA->nuChi0EigscommIndex >= 0) && (!flagNoDmcomm) && (printFlag)) {
        fprintf(outputFile, "global rank = %d, Maximun number of processors for RPA eigenvalue solver is %d\n", rankWorld, pRPA->eig_paral_maxnp);
        fprintf(outputFile, "global rank = %d, The dimension of subgrid for RPA eigen solver is (%d x %d).\n", rankWorld,
                                pRPA->eig_paral_subdims[0], pRPA->eig_paral_subdims[1]);
        fclose(outputFile);
    }
    #endif
    }
    // #else
    // pRPA->eig_useLAPACK = 1;
    // #endif
}


void cyclic_reordering_vectors(int rPermute, int cPermute, int *allCPermute, int size_blacscomm, int *permuteIndices, int *countCPermute) {
    int nuChi0commIndex = 0;
    for (int vecIndex = 0; vecIndex < rPermute; vecIndex++) {
        while (countCPermute[nuChi0commIndex] == allCPermute[nuChi0commIndex]) { // this nuChi0comm is full! cannot contain more vectors
            nuChi0commIndex++;
            nuChi0commIndex = nuChi0commIndex % size_blacscomm; // circulation
        }
        permuteIndices[nuChi0commIndex + countCPermute[nuChi0commIndex]*size_blacscomm] = vecIndex; // vecIndex vector goes to nuChi0commIndex comm
        countCPermute[nuChi0commIndex]++;
        nuChi0commIndex++;
        nuChi0commIndex = nuChi0commIndex % size_blacscomm; // circulation
    }
}


void generate_permute_matrix(int *desc_permute, int *desc_permute_BLCYC, int rPermute, int cPermute, 
    double *permute, double *permute_BLCYC, int ictxt_blacs, MPI_Comm nuChi0BlacsComm) {
    int rank_blackcomm;
    int size_blacscomm;
    MPI_Comm_rank(nuChi0BlacsComm, &rank_blackcomm); // it is nuChi0EigscommIndex
    MPI_Comm_size(nuChi0BlacsComm, &size_blacscomm);
    int *allCPermute = (int*)malloc(sizeof(int)*size_blacscomm);
    MPI_Allgather(&cPermute, 1, MPI_INT, allCPermute, 1, MPI_INT, nuChi0BlacsComm);
    int maxCPermute = allCPermute[0];
    int *permuteIndices = (int*)calloc(sizeof(int), size_blacscomm*maxCPermute);
    int *countCPermute = (int*)calloc(sizeof(int), size_blacscomm);

    cyclic_reordering_vectors(rPermute, cPermute, allCPermute, size_blacscomm, permuteIndices, countCPermute);

    for (int col = 0; col < cPermute; col++) {
        int vecIndex = permuteIndices[rank_blackcomm + col*size_blacscomm];
        permute[vecIndex + col*rPermute] = 1.0;
    }

    int ONE = 1;
    pdgemr2d_(&rPermute, &rPermute, permute, &ONE, &ONE, desc_permute,
        permute_BLCYC, &ONE, &ONE, desc_permute_BLCYC, &ictxt_blacs); 

    free(allCPermute);
    free(permuteIndices);
    free(countCPermute);
}


void generate_permute_matrix_kpt(int *desc_permute, int *desc_permute_BLCYC, int rPermute, int cPermute, 
    double _Complex *permute_kpt, double _Complex *permute_BLCYC_kpt, int ictxt_blacs, MPI_Comm nuChi0BlacsComm) {
    int rank_blackcomm;
    int size_blacscomm;
    MPI_Comm_rank(nuChi0BlacsComm, &rank_blackcomm); // it is nuChi0EigscommIndex
    MPI_Comm_size(nuChi0BlacsComm, &size_blacscomm);
    int *allCPermute = (int*)malloc(sizeof(int)*size_blacscomm);
    MPI_Allgather(&cPermute, 1, MPI_INT, allCPermute, 1, MPI_INT, nuChi0BlacsComm);
    int maxCPermute = allCPermute[0];
    int *permuteIndices = (int*)calloc(sizeof(int), size_blacscomm*maxCPermute);
    int *countCPermute = (int*)calloc(sizeof(int), size_blacscomm);

    cyclic_reordering_vectors(rPermute, cPermute, allCPermute, size_blacscomm, permuteIndices, countCPermute);

    for (int col = 0; col < cPermute; col++) {
        int vecIndex = permuteIndices[rank_blackcomm + col*size_blacscomm];
        permute_kpt[vecIndex + col*rPermute] = 1.0;
    }

    int ONE = 1;
    pzgemr2d_(&rPermute, &rPermute, permute_kpt, &ONE, &ONE, desc_permute,
        permute_BLCYC_kpt, &ONE, &ONE, desc_permute_BLCYC, &ictxt_blacs); 

    free(allCPermute);
    free(permuteIndices);
    free(countCPermute);
}


void setup_HpDiagComm_RPASQ(RPA_OBJ *pRPA, int eig_paral_blksz, int isGammaPoint, int printFlag) {
    int nprocWorld, rankWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    dims_divide_Eigs(nprocWorld, rankWorld, pRPA->nuChi0Neig, &pRPA->npHpDiagSQ);
    pRPA->npHpDiagSQ = judge_npObject(pRPA->nuChi0Neig, nprocWorld, pRPA->npHpDiagSQ);
    if (pRPA->nuChi0Neig / pRPA->npHpDiagSQ < 16) {
        pRPA->npHpDiagSQ = (pRPA->nuChi0Neig / 16) > 0 ? (pRPA->nuChi0Neig / 16) : 1;
    }
    pRPA->nHpDiagComm = distribute_comm_load_LQ(pRPA->nuChi0Neig, pRPA->npHpDiagSQ, rankWorld, nprocWorld, &(pRPA->HpDiagCommIndex), &(pRPA->HpDiagStartIndex), &(pRPA->HpDiagEndIndex));//HAQUE TODO: Copy setup_HpDiagComm_RPASQ to become setup_HpDiagComm_RPALQ. Then, inside setup_HpDiagComm_RPASQ, swap distribute_comm_load_LQ with distribute_comm_load. Then, have a check for places where we call setup_HpDiagComm_RPASQ. If we are actually just using LQ, use setup_HpDiagComm_RPALQ, else, use setup_HpDiagComm_RPASQ.
    int color = (pRPA->HpDiagCommIndex >= 0) ? pRPA->HpDiagCommIndex : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, rankWorld, &pRPA->HpDiagComm);

    pRPA->rank0HpDiagCommInWorld = rankWorld;
    MPI_Bcast(&pRPA->rank0HpDiagCommInWorld, 1, MPI_INT, 0, pRPA->HpDiagComm);
    int nprocHpDiagComm, rankHpDiagComm;
    MPI_Comm_size(pRPA->HpDiagComm, &nprocHpDiagComm);
    MPI_Comm_rank(pRPA->HpDiagComm, &rankHpDiagComm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d HpDiagComm, I will handle HpDiags %d ~ %d\n", rankWorld, pRPA->HpDiagCommIndex, pRPA->HpDiagStartIndex, pRPA->HpDiagEndIndex);
    #endif

    int judgeJoinCompute = (pRPA->HpDiagCommIndex >= 0); // if it is 0, the processor will not join computation
    color = (judgeJoinCompute > 0) ? rankHpDiagComm : INT_MAX;
    MPI_Comm_split(MPI_COMM_WORLD, color, pRPA->HpDiagCommIndex, &pRPA->HpDiagBridgeComm);
    pRPA->HpDiagBridgeCommIndex = (judgeJoinCompute > 0) ? rankHpDiagComm : -1;
    int rankHpDiagBridgeComm; 
    MPI_Comm_rank(pRPA->HpDiagBridgeComm, &rankHpDiagBridgeComm);
    #ifdef DEBUG
    printf("I am %d in comm world, I am in %d HpDiagBridgeComm, My rank in it is %d\n", rankWorld, pRPA->HpDiagBridgeCommIndex, rankHpDiagBridgeComm);
    #endif

    color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rankWorld, &pRPA->commWorld);

    FILE *outputFile = NULL;
    #ifdef DEBUG
    if ((pRPA->HpDiagCommIndex >= 0) && (printFlag)) {
        outputFile = fopen(pRPA->timeRecordname, "a");
    }
    #endif
    int dims[3] = {0, 0, 0}; 
    // int HpSizes[2];
    // HpSizes[0] = pRPA->nuChi0Neig;
    // HpSizes[1] = pRPA->nuChi0Neig;
    // ScaLAPACK_Dims_2D_BLCYC(nprocHpDiagComm, HpSizes, dims);
    dims[0] = nprocHpDiagComm;
    dims[1] = 1; // just divide symmetric matrix Hp in rows, to make matrix vector multiplication easily: Hp^T*vk
    int *usermap;
    Cblacs_get( -1, 0, &pRPA->ictxt_HpSQ );
    if (pRPA->HpDiagCommIndex >= 0) {
        usermap = (int *)malloc(sizeof(int)*nprocHpDiagComm);
        // create usermap_1 = reshape(usermap,[dims[0], dims[1]])
        for (int i = 0; i < nprocHpDiagComm; i++) {
            usermap[i] = pRPA->rank0HpDiagCommInWorld + i;
        }
    } else {
        usermap = (int *)malloc(sizeof(int)*1);
        usermap[0] = rankWorld;
        dims[0] = dims[1] = 1;
    }
    Cblacs_gridmap( &pRPA->ictxt_HpSQ, usermap, dims[0], dims[0], dims[1] ); // Cart topology
    free(usermap);
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo( pRPA->ictxt_HpSQ, &nprow, &npcol, &myrow, &mycol );

    int ZERO = 0; int mbHpSQ, nbHpSQ, lldaHpSQ, info; // block size for storing Hp and Mp
    if (nprocHpDiagComm == 1) { // in this case we will call LAPACK instead to solve the subspace eigenproblem
        mbHpSQ = nbHpSQ = pRPA->nuChi0Neig;
    } else { // in this case we will use ScaLAPACK to solve the subspace eigenproblem
        mbHpSQ = eig_paral_blksz;
        nbHpSQ = pRPA->nuChi0Neig;
    }
    if (pRPA->HpDiagCommIndex >= 0) {
        pRPA->nrHpSQ = numroc_( &pRPA->nuChi0Neig, &mbHpSQ, &myrow, &ZERO, &nprow);
        pRPA->nrHpSQ = max(1, pRPA->nrHpSQ);
        pRPA->ncHpSQ = numroc_( &pRPA->nuChi0Neig, &nbHpSQ, &mycol, &ZERO, &npcol);
        pRPA->ncHpSQ = max(1, pRPA->ncHpSQ);
    } else {
        pRPA->nrHpSQ = pRPA->ncHpSQ = 1;
    }
    lldaHpSQ = max(1, pRPA->nrHpSQ);
    if (pRPA->HpDiagCommIndex >= 0) {
        descinit_(&pRPA->desc_HpSQ[0], &pRPA->nuChi0Neig, &pRPA->nuChi0Neig,
                  &mbHpSQ, &nbHpSQ, &ZERO, &ZERO, &pRPA->ictxt_HpSQ, &lldaHpSQ, &info);
    } else {
        for (int i = 0; i < 9; i++) {
            pRPA->desc_HpSQ[0] = 0;
        }
    }
    #ifdef DEBUG
    if ((pRPA->HpDiagCommIndex >= 0) && (printFlag)) {
        fprintf(outputFile, "global rank = %d, HpSQ topo %d, mbHpSQ = %d, myrow %d, nprow %d, mycol %d, npcol %d, nrHpSQ = %d, ncHpSQ = %d\n", rankWorld, pRPA->HpDiagCommIndex, mbHpSQ, myrow, nprow, mycol, npcol, pRPA->nrHpSQ, pRPA->ncHpSQ);
    }
    #endif
    if (isGammaPoint) {
        pRPA->HpSQ = (double *)malloc(pRPA->nrHpSQ * pRPA->ncHpSQ * sizeof(double));
        assert(pRPA->HpSQ);
    } else {
        pRPA->HpSQ_kpt = (double _Complex*)malloc(pRPA->nrHpSQ * pRPA->ncHpSQ * sizeof(double _Complex));
        assert(pRPA->HpSQ_kpt);
    }

    Cblacs_get( -1, 0, &pRPA->ctxtWorld);
    int *globalMap = (int*)malloc(sizeof(int) * nprocWorld);
    for (int i = 0; i < nprocWorld; i++) {
        globalMap[i] = i;
    }
    Cblacs_gridmap( &pRPA->ctxtWorld, globalMap, 1, 1, nprocWorld );
    free(globalMap);

    int mbvk, nbvk, lldavk;
    if (nprocHpDiagComm == 1) { // in this case we will call LAPACK
        mbvk = pRPA->nuChi0Neig;
    } else { // in this case we will use ScaLAPACK
        mbvk = pRPA->nuChi0Neig / nprocHpDiagComm + ((pRPA->nuChi0Neig % nprocHpDiagComm > 0) ? 1 : 0);
    }
    pRPA->vkStart = mbvk * rankHpDiagComm;
    if (pRPA->vkStart > pRPA->nuChi0Neig) {
        pRPA->vkStart = pRPA->nuChi0Neig;
    }
    if (pRPA->HpDiagCommIndex >= 0) {
        pRPA->nrvk = numroc_( &pRPA->nuChi0Neig, &mbvk, &myrow, &ZERO, &nprow);
        pRPA->nrvk = max(1, pRPA->nrvk);
    } else {
        pRPA->nrvk = 1;
    }
    nbvk = pRPA->nHpDiagComm;
    
    if (pRPA->vkStart < pRPA->nuChi0Neig) {
        pRPA->vkEnd = pRPA->vkStart + pRPA->nrvk - 1;
    } else {
        pRPA->vkEnd = pRPA->vkStart;
    }
    
    lldavk = max(1, pRPA->nrvk);
    if (pRPA->HpDiagCommIndex >= 0) {
        descinit_(&pRPA->desc_vk[0], &pRPA->nuChi0Neig, &pRPA->nHpDiagComm,
                  &mbvk, &nbvk, &ZERO, &ZERO, &pRPA->ictxt_HpSQ, &lldavk, &info);
    } else {
        for (int i = 0; i < 9; i++) {
            pRPA->desc_vk[0] = 0;
        }
    }

    int mbvk_BLCYC, nbvk_BLCYC, lldavk_BLCYC;
    if (nprocHpDiagComm == 1) { // in this case we will call LAPACK
        mbvk_BLCYC = pRPA->nuChi0Neig;
        nbvk_BLCYC = pRPA->nHpDiagComm;
    } else { // in this case we will use ScaLAPACK
        mbvk_BLCYC = eig_paral_blksz;
        nbvk_BLCYC = pRPA->nHpDiagComm;
    }
    
    if (pRPA->HpDiagCommIndex >= 0) {
        pRPA->nrvk_BLCYC = numroc_( &pRPA->nuChi0Neig, &mbvk_BLCYC, &myrow, &ZERO, &nprow);
        pRPA->nrvk_BLCYC = max(1, pRPA->nrvk_BLCYC);
    } else {
        pRPA->nrvk_BLCYC = 1;
    }
    lldavk_BLCYC = max(1, pRPA->nrvk_BLCYC);
    if (pRPA->HpDiagCommIndex >= 0) {
        descinit_(&pRPA->desc_vk_BLCYC[0], &pRPA->nuChi0Neig, &pRPA->nHpDiagComm,
                  &mbvk_BLCYC, &nbvk_BLCYC, &ZERO, &ZERO, &pRPA->ictxt_HpSQ, &lldavk_BLCYC, &info);
    } else {
        for (int i = 0; i < 9; i++) {
            pRPA->desc_vk_BLCYC[0] = 0;
        }
    }

    #ifdef DEBUG
    if ((pRPA->HpDiagCommIndex >= 0) && (printFlag)) {
        fprintf(outputFile, "global rank = %d, HpSQ topo %d, myrow %d, nprow %d, mycol %d, npcol %d, nrvk %d, vkStart %d, vkEnd %d, nrvk_BLCYC %d\n", rankWorld, pRPA->HpDiagCommIndex, myrow, nprow, mycol, npcol,
            pRPA->nrvk, pRPA->vkStart, pRPA->vkEnd, pRPA->nrvk_BLCYC);
        fclose(outputFile);
    }
    #endif
}

void generate_permute_matrix_SQ(int *desc_permute_BLCYC, int sizePermute, int rPermute_BLCYC, int cPermute_BLCYC, 
    double *permute_BLCYC, int *desc_pTp, double *pTp, int ictxt_blacs, MPI_Comm nuChi0BlacsComm) { // generate a random rotation matrix
    int rank_blacscomm;
    int size_blacscomm;
    MPI_Comm_rank(nuChi0BlacsComm, &rank_blacscomm); // it is nuChi0EigscommIndex
    MPI_Comm_size(nuChi0BlacsComm, &size_blacscomm);
    double *randInitPermute_BLCYC = (double*)malloc(sizeof(double)*rPermute_BLCYC*cPermute_BLCYC);
    
    int seed = rank_blacscomm + 42, ONE = 1; double alpha = 1.0, beta = 0.0;
    SetRandMat_seed(randInitPermute_BLCYC, rPermute_BLCYC, cPermute_BLCYC, -0.5, 0.5, seed);
    
    memcpy(permute_BLCYC, randInitPermute_BLCYC, sizeof(double)*rPermute_BLCYC*cPermute_BLCYC);
    
    pdgemm_("T", "N", &sizePermute, &sizePermute, &sizePermute, &alpha, 
                permute_BLCYC, &ONE, &ONE, desc_permute_BLCYC, 
                randInitPermute_BLCYC, &ONE, &ONE, desc_permute_BLCYC, 
                &beta, pTp, &ONE, &ONE, desc_pTp);
    
    Chol_orth(permute_BLCYC, desc_permute_BLCYC, pTp, desc_pTp, &sizePermute, &sizePermute);

    free(randInitPermute_BLCYC);
}