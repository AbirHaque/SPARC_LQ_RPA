/**
 * @file    parallelization_SPARC.c
 * @brief   This file contains the functions related to MPI comm setup and vector/matrix initialization.
 *          The parallelization levels hold by pSPARC is k-point and band. These comms are splitted from
 *          nuChi0 comm.
 *
 * @authors This file is modified from file parallelization.c of SPARC
 *          Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#define _XOPEN_SOURCE 500 // For srand48(), drand48(), usleep()

// For sched_setaffinity
#define  _GNU_SOURCE
#include <sched.h>
#include <unistd.h>  // Also for usleep()

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <complex.h>
/* ScaLAPACK routines */
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

#include "parallelization.h"
#include "tools.h"
#include "initialization.h"
#include "electrostatics.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

/**
 * @brief   Set up sub-communicators in nuChi0Eigscomm.
 * two things to do: 
 * 1. read the npspin, npkpt, npband and npNd again from the input
 * 2. free all allocated space at the first initialization before running this function
 */
void Setup_Comms_SPARC(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld) {
    int i, j, dims[3] = {0, 0, 0}, periods[3], ierr;
    int nproc, rank;
    int size_spincomm, rank_spincomm;
    int nproc_kptcomm, rank_kptcomm, size_kptcomm;
    int nproc_bandcomm, rank_bandcomm, size_bandcomm, NP_BANDCOMM, NB;
    int npNd, gridsizes[3], minsize, coord_dmcomm[3], rank_dmcomm;
    int color;
#ifdef DEBUG
    double t1, t2;
#endif
    MPI_Comm_size(nuChi0Eigscomm, &nproc);
    MPI_Comm_rank(nuChi0Eigscomm, &rank);

    int flagPrintNuChi0Eigscomm = (nuChi0EigscommIndex == 0); // set which nuchi0Eigscomm will print output
#ifdef DEBUG
    if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("Set up communicators.\n");
#endif

    // in the case user doesn't set any parallelization parameters, search for a good
    // combination in advance
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
    minsize = pSPARC->order/2;
    if (pSPARC->npspin == 0 && pSPARC->npkpt == 0 && 
        pSPARC->npband == 0 && npNd == 0) 
    {
        dims_divide_skbd(pSPARC->Nspin, pSPARC->Nkpts_sym, pSPARC->Nstates, 
            gridsizes, nproc, &pSPARC->npspin, &pSPARC->npkpt, &pSPARC->npband, &npNd, minsize, pSPARC->usefock);
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx = dims[0];
        pSPARC->npNdy = dims[1];
        pSPARC->npNdz = dims[2];
    }

    //------------------------------------------------//
    //                 set up spincomm                //
    //------------------------------------------------//
    // Allocate number of spin communicators
    // if npspin is not provided by user, or the given npspin is too large
    if (pSPARC->npspin == 0) {
        pSPARC->npspin = min(nproc, pSPARC->Nspin);
    } else if (pSPARC->npspin > pSPARC->Nspin || pSPARC->npspin > nproc) {
        pSPARC->npspin = min(nproc, pSPARC->Nspin);
        if ((rank == 0) && flagPrintNuChi0Eigscomm) {
            printf("WARNING: npspin is larger than pSPARC->Nspin or nproc!\n"
                   "         Forcing npspin = min(nproc, pSPARC->Nspin) = %d.\n\n",pSPARC->npspin);
        }
    }

    // Allocate number of processors and their local indices in a spin communicator
    size_spincomm = nproc / pSPARC->npspin;
    if (rank < (nproc - nproc % pSPARC->npspin))
        pSPARC->spincomm_index = rank / size_spincomm;
    else
        pSPARC->spincomm_index = -1;

    // calculate number of spin assigned to each spincomm
    if (rank < (nproc - nproc % pSPARC->npspin)) {
        pSPARC->Nspin_spincomm = pSPARC->Nspin / pSPARC->npspin;
        pSPARC->Nspinor_spincomm = pSPARC->Nspinor / pSPARC->npspin;
    } else {
        pSPARC->Nspin_spincomm = 0;
        pSPARC->Nspinor_spincomm = 0;
    }

    // calculate start and end indices of the spin obtained by each spincomm
    if (pSPARC->spincomm_index == -1) {
        pSPARC->spin_start_indx = 0;
        pSPARC->spinor_start_indx = 0;
    } else {
        pSPARC->spin_start_indx = pSPARC->spincomm_index * pSPARC->Nspin_spincomm;
        pSPARC->spinor_start_indx = pSPARC->spincomm_index * pSPARC->Nspinor_spincomm;
    }
    pSPARC->spin_end_indx = pSPARC->spin_start_indx + pSPARC->Nspin_spincomm - 1;
    pSPARC->spinor_end_indx = pSPARC->spinor_start_indx + pSPARC->Nspinor_spincomm - 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the nuChi0Eigscomm into spincomms using color = spincomm_index
    color = (pSPARC->spincomm_index >= 0) ? pSPARC->spincomm_index : INT_MAX; 
    MPI_Comm_split(nuChi0Eigscomm, color, 0, &pSPARC->spincomm);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("\n--set up spincomm took %.3f ms\n",(t2-t1)*1000);
#endif
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);

    //------------------------------------------------------//
    //              set up spin_bridge_comm                  //
    //------------------------------------------------------//
    // spin_bridge_comm contains all processes with the same rank_spincomm
    if (rank < nproc - nproc % pSPARC->npspin) {
        color = rank_spincomm;
    } else {
        color = INT_MAX;
    }
    MPI_Comm_split(nuChi0Eigscomm, color, pSPARC->spincomm_index, &pSPARC->spin_bridge_comm);

    //------------------------------------------------//
    //                 set up kptcomm                 //
    //------------------------------------------------//
    // if npkpt is not provided by user, or the given npkpt is too large
    if (pSPARC->npkpt == 0) {
        pSPARC->npkpt = min(size_spincomm, pSPARC->Nkpts_sym); // paral over k-points as much as possible
    } else if (pSPARC->npkpt > pSPARC->Nkpts_sym || pSPARC->npkpt > size_spincomm) {
        pSPARC->npkpt = min(size_spincomm, pSPARC->Nkpts_sym);
        if ((rank == 0) && flagPrintNuChi0Eigscomm) {
            printf("WARNING: npkpt is larger than number of k-points after symmetry reduction or size_spincomm!\n"
                   "         Forcing npkpt = min(size_spincomm, Nkpts_sym) = %d.\n\n",pSPARC->npkpt);
        }
    }

    size_kptcomm = size_spincomm / pSPARC->npkpt; // size of kptcomm
    if (pSPARC->spincomm_index != -1 && rank_spincomm < (size_spincomm - size_spincomm % pSPARC->npkpt))
        pSPARC->kptcomm_index = rank_spincomm / size_kptcomm;
    else
        pSPARC->kptcomm_index = -1;

    // calculate number of k-points assigned to each kptcomm
    if (rank_spincomm < (size_spincomm - size_spincomm % pSPARC->npkpt))
        pSPARC->Nkpts_kptcomm = pSPARC->Nkpts_sym / pSPARC->npkpt + (int) (pSPARC->kptcomm_index < (pSPARC->Nkpts_sym % pSPARC->npkpt));
    else
        pSPARC->Nkpts_kptcomm = 0;

    // calculate start and end indices of the k-points obtained by each kptcomm
    if (pSPARC->kptcomm_index == -1) {
        pSPARC->kpt_start_indx = 0;
    } else if (pSPARC->kptcomm_index < (pSPARC->Nkpts_sym % pSPARC->npkpt)) {
        pSPARC->kpt_start_indx = pSPARC->kptcomm_index * pSPARC->Nkpts_kptcomm;
    } else {
        pSPARC->kpt_start_indx = pSPARC->kptcomm_index * pSPARC->Nkpts_kptcomm + pSPARC->Nkpts_sym % pSPARC->npkpt;
    }
    pSPARC->kpt_end_indx = pSPARC->kpt_start_indx + pSPARC->Nkpts_kptcomm - 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the pSPARC->spincomm into several kptcomms using color = kptcomm_index
    color = (pSPARC->kptcomm_index >= 0) ? pSPARC->kptcomm_index : INT_MAX;
    MPI_Comm_split(pSPARC->spincomm, color, 0, &pSPARC->kptcomm);

    //setup_core_affinity(pSPARC->kptcomm);
    //bind_proc_to_phys_core();

#ifdef DEBUG
    t2 = MPI_Wtime();
    if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("\n--set up kptcomm took %.3f ms\n",(t2-t1)*1000);
#endif

    // Local k-points array
    if (pSPARC->Nkpts >= 1 && pSPARC->kptcomm_index != -1) {
        // allocate memory for storing local k-points and the weights for k-points
        //if (pSPARC->BC != 1) {
            pSPARC->kptWts_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            pSPARC->k1_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            pSPARC->k2_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            pSPARC->k3_loc = (double *)malloc(pSPARC->Nkpts_kptcomm * sizeof(double));
            // calculate the k point weights
            Calculate_local_kpoints(pSPARC);
        //}
    }

    //-------------------------------------------------------//
    //    set up a Cartesian topology within each kptcomm    //
    //-------------------------------------------------------//
    MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kptcomm);
    if (pSPARC->kptcomm_index < 0) nproc_kptcomm = size_kptcomm; // let all proc know the size

    /* first calculate the best dimensions using SPARC_Dims_create */
    // gridsizes = {Nx, Ny, Nz}, minsize, and periods are the same as above
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
    minsize = pSPARC->order/2;

    // calculate dims[]
    SPARC_Dims_create(nproc_kptcomm, 3, gridsizes, minsize, dims, &ierr);

    pSPARC->npNdx_kptcomm = dims[0];
    pSPARC->npNdy_kptcomm = dims[1];
    pSPARC->npNdz_kptcomm = dims[2];

#ifdef DEBUG
    if ((!rank) && flagPrintNuChi0Eigscomm)
    printf("\n kpt_topo #%d, kptcomm topology dims = {%d, %d, %d}, nodes/proc = {%.2f,%.2f,%.2f}\n", pSPARC->kptcomm_index,
            dims[0],dims[1],dims[2],(double)gridsizes[0]/dims[0],(double)gridsizes[1]/dims[1],(double)gridsizes[2]/dims[2]);
#endif

    if (pSPARC->kptcomm_index >= 0) {
        //Processes in kptcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
        MPI_Cart_create(pSPARC->kptcomm, 3, dims, periods, 1, &pSPARC->kptcomm_topo); // 1 is to reorder rank
    } else {
        pSPARC->kptcomm_topo = MPI_COMM_NULL;
    }

    int rank_kpt_topo, coord_kpt_topo[3];
    if (pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        MPI_Comm_rank(pSPARC->kptcomm_topo, &rank_kpt_topo);
        MPI_Cart_coords(pSPARC->kptcomm_topo, rank_kpt_topo, 3, coord_kpt_topo);

        // find size of distributed domain over kptcomm_topo
        pSPARC->Nx_d_kptcomm = block_decompose(gridsizes[0], dims[0], coord_kpt_topo[0]);
        pSPARC->Ny_d_kptcomm = block_decompose(gridsizes[1], dims[1], coord_kpt_topo[1]);
        pSPARC->Nz_d_kptcomm = block_decompose(gridsizes[2], dims[2], coord_kpt_topo[2]);
        pSPARC->Nd_d_kptcomm = pSPARC->Nx_d_kptcomm * pSPARC->Ny_d_kptcomm * pSPARC->Nz_d_kptcomm;

        // find corners of the distributed domain in kptcomm_topo
        pSPARC->DMVertices_kptcomm[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_kpt_topo[0]);
        pSPARC->DMVertices_kptcomm[1] = pSPARC->DMVertices_kptcomm[0] + pSPARC->Nx_d_kptcomm - 1;
        pSPARC->DMVertices_kptcomm[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_kpt_topo[1]);
        pSPARC->DMVertices_kptcomm[3] = pSPARC->DMVertices_kptcomm[2] + pSPARC->Ny_d_kptcomm - 1;
        pSPARC->DMVertices_kptcomm[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_kpt_topo[2]);
        pSPARC->DMVertices_kptcomm[5] = pSPARC->DMVertices_kptcomm[4] + pSPARC->Nz_d_kptcomm - 1;
    } else {
        rank_kpt_topo = -1;
        coord_kpt_topo[0] = -1; coord_kpt_topo[1] = -1; coord_kpt_topo[2] = -1;
        pSPARC->Nx_d_kptcomm = 0;
        pSPARC->Ny_d_kptcomm = 0;
        pSPARC->Nz_d_kptcomm = 0;
        pSPARC->Nd_d_kptcomm = 0;
        pSPARC->DMVertices_kptcomm[0] = 0;
        pSPARC->DMVertices_kptcomm[1] = 0;
        pSPARC->DMVertices_kptcomm[2] = 0;
        pSPARC->DMVertices_kptcomm[3] = 0;
        pSPARC->DMVertices_kptcomm[4] = 0;
        pSPARC->DMVertices_kptcomm[5] = 0;
        //pSPARC->Nband_bandcomm = 0;
    }

    if(pSPARC->cell_typ != 0) {
        if(pSPARC->kptcomm_topo != MPI_COMM_NULL) {
            int rank_chk, dir, tmp;
            int ncoords[3];
            int nnproc = 1; // No. of neighboring processors in each direction
            int nneighb = pow(2*nnproc+1,3) - 1; // 6 faces + 12 edges + 8 corners
            int *neighb;
            neighb = (int *) malloc(nneighb * sizeof (int));
            int count = 0, i, j, k;
            for(k = -nnproc; k <= nnproc; k++){
                for(j = -nnproc; j <= nnproc; j++){
                    for(i = -nnproc; i <= nnproc; i++){
                        tmp = 0;
                        if(i == 0 && j == 0 && k == 0){
                            continue;
                        } else{
                            ncoords[0] = coord_kpt_topo[0] + i;
                            ncoords[1] = coord_kpt_topo[1] + j;
                            ncoords[2] = coord_kpt_topo[2] + k;
                            for(dir = 0; dir < 3; dir++){
                                if(periods[dir]){
                                    if(ncoords[dir] < 0)
                                        ncoords[dir] += dims[dir];
                                    else if(ncoords[dir] >= dims[dir])
                                        ncoords[dir] -= dims[dir];
                                    tmp = 1;
                                } else{
                                    if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                        rank_chk = MPI_PROC_NULL;
                                        tmp = 0;
                                        break;
                                    }
                                    else
                                        tmp = 1;
                                }
                            }
                            //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                            if(tmp == 1)
                                MPI_Cart_rank(pSPARC->kptcomm_topo,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                            neighb[count] = rank_chk;
                            count++;
                        }
                    }
                }
            }
            MPI_Dist_graph_create_adjacent(pSPARC->kptcomm_topo,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,&pSPARC->kptcomm_topo_dist_graph);
            free(neighb);
        }
    }

    //------------------------------------------------------------------------------------//
    //    set up inter-communicators between the Cart topology and the rest in kptcomm    //
    //------------------------------------------------------------------------------------//
    int nproc_kptcomm_topo;
    nproc_kptcomm_topo = pSPARC->npNdx_kptcomm * pSPARC->npNdy_kptcomm * pSPARC->npNdz_kptcomm;
    if (nproc_kptcomm_topo < nproc_kptcomm && pSPARC->kptcomm_index >= 0) {
#ifdef DEBUG
        t1 = MPI_Wtime();
#endif
        // first create a comm that includes all the processes that are excluded from the Cart topology
        MPI_Group kptgroup, kptgroup_excl;
        MPI_Comm_group(pSPARC->kptcomm, &kptgroup);
        int *incl_ranks, count;
        incl_ranks = (int *)malloc((nproc_kptcomm - nproc_kptcomm_topo) * sizeof(int));
        count = 0;
        for (i = nproc_kptcomm_topo; i < nproc_kptcomm; i++) {
            incl_ranks[count] = i; count++;
        }
        MPI_Group_incl(kptgroup, count, incl_ranks, &kptgroup_excl);
        MPI_Comm_create_group(pSPARC->kptcomm, kptgroup_excl, 110, &pSPARC->kptcomm_topo_excl);

        // now create an inter-comm between kptcomm_topo and kptcomm_topo_excl
        if (pSPARC->kptcomm_topo != MPI_COMM_NULL) {
            MPI_Intercomm_create(pSPARC->kptcomm_topo, 0, pSPARC->kptcomm, nproc_kptcomm_topo, 111, &pSPARC->kptcomm_inter);
        } else {
            MPI_Intercomm_create(pSPARC->kptcomm_topo_excl, 0, pSPARC->kptcomm, 0, 111, &pSPARC->kptcomm_inter);
        }

        free(incl_ranks);
        MPI_Group_free(&kptgroup);
        MPI_Group_free(&kptgroup_excl);

#ifdef DEBUG
        t2 = MPI_Wtime();
        if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("\n--set up kptcomm_inter took %.3f ms\n" ,(t2-t1)*1000);
#endif
    } else {
        pSPARC->kptcomm_topo_excl = MPI_COMM_NULL;
        pSPARC->kptcomm_inter = MPI_COMM_NULL;
    }

    //------------------------------------------------------//
    //              set up kpt_bridge_comm                  //
    //------------------------------------------------------//
    // kpt_bridge_comm contains all processes with the same rank_kptcomm
    if (rank_spincomm < size_spincomm - size_spincomm % pSPARC->npkpt) {
        color = rank_kptcomm;
    } else {
        color = INT_MAX;
    }
    MPI_Comm_split(pSPARC->spincomm, color, pSPARC->kptcomm_index, &pSPARC->kpt_bridge_comm); // TODO: exclude null kptcomms

    //------------------------------------------------//
    //               set up bandcomm                  //
    //------------------------------------------------//
    // in each kptcomm, create sub-communicators over bands
    // note: different from kptcomms, we require the bandcomms in the same kptcomm to be of the same size
    if (pSPARC->npband == 0) {
        pSPARC->npband = min(nproc_kptcomm, pSPARC->Nstates); // paral over band as much as possible
    } else if (pSPARC->npband > nproc_kptcomm) { 
        // here we allow user to give npband larger than Nstates!
        pSPARC->npband = min(nproc_kptcomm, pSPARC->Nstates);
    }

    size_bandcomm = nproc_kptcomm / pSPARC->npband; // size of each bandcomm
    NP_BANDCOMM = pSPARC->npband * size_bandcomm; // number of processors that belong to a bandcomm, others are excluded
    // calculate which bandcomm processor with RANK = rank_kptcomm belongs to
    if (rank_kptcomm < NP_BANDCOMM && pSPARC->kptcomm_index != -1) {
        //pSPARC->bandcomm_index = rank_kptcomm / size_bandcomm; // assign processes column-wisely
        pSPARC->bandcomm_index = rank_kptcomm % pSPARC->npband; // assign processes row-wisely
    } else {
        pSPARC->bandcomm_index = -1; // these processors won't be used to do calculations
    }
    // calculate number of bands assigned to each bandcomm, this is a special case of block-cyclic distribution (*,block)
    if (pSPARC->bandcomm_index == -1) {
        pSPARC->Nband_bandcomm = 0;
    } else {
        NB = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // this is equal to ceil(Nstates/npband), for int inputs only
        pSPARC->Nband_bandcomm = pSPARC->bandcomm_index < (pSPARC->Nstates / NB) ? NB : (pSPARC->bandcomm_index == (pSPARC->Nstates / NB) ? (pSPARC->Nstates % NB) : 0);
    }

    // calculate start and end indices of the bands obtained by each kptcomm
    if (pSPARC->bandcomm_index == -1) {
        pSPARC->band_start_indx = 0;
    } else if (pSPARC->bandcomm_index <= (pSPARC->Nstates / NB)) {
        pSPARC->band_start_indx = pSPARC->bandcomm_index * NB;
    } else {
        pSPARC->band_start_indx = pSPARC->Nstates; // TODO: this might be dangerous, consider using 0, instead of Ns here
    }
    pSPARC->band_end_indx = pSPARC->band_start_indx + pSPARC->Nband_bandcomm - 1;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the kptcomm into several bandcomms using color = bandcomm_index
    color = (pSPARC->bandcomm_index >= 0) ? pSPARC->bandcomm_index : INT_MAX;
    MPI_Comm_split(pSPARC->kptcomm, color, 0, &pSPARC->bandcomm);

#ifdef DEBUG
    t2 = MPI_Wtime();
    if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("\n--set up bandcomm took %.3f ms\n",(t2-t1)*1000);
#endif

    //------------------------------------------------//
    //               set up domaincomm                //
    //------------------------------------------------//
    MPI_Comm_size(pSPARC->bandcomm, &nproc_bandcomm);
    MPI_Comm_rank(pSPARC->bandcomm, &rank_bandcomm);
    // let all proc know the size of bandcomm
    if (pSPARC->bandcomm_index < 0) nproc_bandcomm = size_bandcomm; 

    npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    minsize = pSPARC->order/2;

    if (npNd == 0)  {
        // when user does not provide domain decomposition parameters
        npNd = nproc_bandcomm;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx = dims[0];
        pSPARC->npNdy = dims[1];
        pSPARC->npNdz = dims[2];
    } else if (npNd < 0 || npNd > nproc_bandcomm || pSPARC->Nx / pSPARC->npNdx < minsize ||
               pSPARC->Ny / pSPARC->npNdy < minsize || pSPARC->Nz / pSPARC->npNdz < minsize) {
        // when domain decomposition parameters are not reasonable
        npNd = nproc_bandcomm;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        pSPARC->npNdx = dims[0];
        pSPARC->npNdy = dims[1];
        pSPARC->npNdz = dims[2];
    } else {
        dims[0] = pSPARC->npNdx;
        dims[1] = pSPARC->npNdy;
        dims[2] = pSPARC->npNdz;
    }

    // recalculate number of processors in each dmcomm
    npNd = pSPARC->npNdx * pSPARC->npNdy * pSPARC->npNdz;
    pSPARC->npNd = npNd;

    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;
#ifdef DEBUG
    if ((!rank) && flagPrintNuChi0Eigscomm) printf("rank = %d, dmcomm dims = {%d, %d, %d}\n", rank, pSPARC->npNdx, pSPARC->npNdy, pSPARC->npNdz);
    t1 = MPI_Wtime();
#endif

    // Processes in bandcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
    if (pSPARC->bandcomm_index != -1) {
        MPI_Cart_create(pSPARC->bandcomm, 3, dims, periods, 1, &pSPARC->dmcomm); // 1 is to reorder rank
    } else {
        pSPARC->dmcomm = MPI_COMM_NULL;
    }

#ifdef DEBUG
    if ((!rank) && flagPrintNuChi0Eigscomm) printf("gridsizes = [%d, %d, %d], Nstates = %d, dmcomm dims = [%d, %d, %d]\n",
        gridsizes[0],gridsizes[1],gridsizes[2],pSPARC->Nstates,dims[0],dims[1],dims[2]);
#endif

    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index != -1) {
        MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
        MPI_Cart_coords(pSPARC->dmcomm, rank_dmcomm, 3, coord_dmcomm);

        // find size of distributed domain over dmcomm
        pSPARC->Nx_d_dmcomm = block_decompose(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->Ny_d_dmcomm = block_decompose(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->Nz_d_dmcomm = block_decompose(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->Nd_d_dmcomm = pSPARC->Nx_d_dmcomm * pSPARC->Ny_d_dmcomm * pSPARC->Nz_d_dmcomm;

        // find corners of the distributed domain in dmcomm
        pSPARC->DMVertices_dmcomm[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->DMVertices_dmcomm[1] = pSPARC->DMVertices_dmcomm[0] + pSPARC->Nx_d_dmcomm - 1;
        pSPARC->DMVertices_dmcomm[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->DMVertices_dmcomm[3] = pSPARC->DMVertices_dmcomm[2] + pSPARC->Ny_d_dmcomm - 1;
        pSPARC->DMVertices_dmcomm[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->DMVertices_dmcomm[5] = pSPARC->DMVertices_dmcomm[4] + pSPARC->Nz_d_dmcomm - 1;

    } else {
        rank_dmcomm = -1;
        coord_dmcomm[0] = -1; coord_dmcomm[1] = -1; coord_dmcomm[2] = -1;
        pSPARC->Nx_d_dmcomm = 0;
        pSPARC->Ny_d_dmcomm = 0;
        pSPARC->Nz_d_dmcomm = 0;
        pSPARC->Nd_d_dmcomm = 0;
        pSPARC->DMVertices_dmcomm[0] = 0;
        pSPARC->DMVertices_dmcomm[1] = 0;
        pSPARC->DMVertices_dmcomm[2] = 0;
        pSPARC->DMVertices_dmcomm[3] = 0;
        pSPARC->DMVertices_dmcomm[4] = 0;
        pSPARC->DMVertices_dmcomm[5] = 0;
        //pSPARC->Nband_bandcomm = 0;
    }
#ifdef DEBUG
    t2 = MPI_Wtime();
    if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("\n--set up dmcomm took %.3f ms\n",(t2-t1)*1000);
#endif

    // Set up a 26 neighbor communicator for nonorthogonal systems
    // TODO: Modify the communicator based on number of non zero enteries in off diagonal of pSPARC->lapcT
    // TODO: Take into account more than 26 Neighbors
    if(pSPARC->cell_typ != 0) {
        if(pSPARC->dmcomm != MPI_COMM_NULL) {
            int rank_chk, dir, tmp;
            int ncoords[3];
            int nnproc = 1; // No. of neighboring processors in each direction
            int nneighb = pow(2*nnproc+1,3) - 1; // 6 faces + 12 edges + 8 corners
            int *neighb;
            neighb = (int *) malloc(nneighb * sizeof (int));
            int count = 0, i, j, k;

            for(k = -nnproc; k <= nnproc; k++){
                for(j = -nnproc; j <= nnproc; j++){
                    for(i = -nnproc; i <= nnproc; i++){
                        tmp = 0;
                        if(i == 0 && j == 0 && k == 0){
                            continue;
                        } else{
                            ncoords[0] = coord_dmcomm[0] + i;
                            ncoords[1] = coord_dmcomm[1] + j;
                            ncoords[2] = coord_dmcomm[2] + k;
                            for(dir = 0; dir < 3; dir++){
                                if(periods[dir]){
                                    if(ncoords[dir] < 0)
                                        ncoords[dir] += dims[dir];
                                    else if(ncoords[dir] >= dims[dir])
                                        ncoords[dir] -= dims[dir];
                                    tmp = 1;
                                } else{
                                    if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                        rank_chk = MPI_PROC_NULL;
                                        tmp = 0;
                                        break;
                                    }
                                    else
                                        tmp = 1;
                                }
                            }
                            //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                            if(tmp == 1)
                                MPI_Cart_rank(pSPARC->dmcomm,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                            neighb[count] = rank_chk;
                            count++;
                        }
                    }
                }
            }
            MPI_Dist_graph_create_adjacent(pSPARC->dmcomm,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,&pSPARC->comm_dist_graph_psi); // creates a distributed graph topology (adjacent, cartesian cubical)
            //pSPARC->dmcomm_phi = pSPARC->comm_dist_graph_phi;
            free(neighb);
        }
    }

    //------------------------------------------------//
    //                set up blacscomm                //
    //------------------------------------------------//
    // There are two ways to set up the blacscomm:
    // 1. Let the group of all processors dealing with the same local domain (but different
    //    bands) be a blacscomm. We will find Psi_i' * Psi_i in each blacscomm, and sum
    //    over all local domains (dmcomm) to find Psi' * Psi. This way, however, doesn't
    //    seem to be efficient since the sum (Allreduce) part will take a lot of time. In
    //    some cases even more than the computation time.
    // 2. In stead of having many parallel blacscomm in each kptcomm, we create only one
    //    blacscomm, containing all the parallel blacscomm described above. This way
    //    uses a little trick of linear algebra: since the rows are reordered when we
    //    do domain parallelization, the matrix Psi is interpretated as P * Psi by the
    //    ScaLAPACK routines distributed block-wisely, where P is a permutation matrix
    //    (with 0's and 1's). But, since P is orthogonal, we could still preceed and find
    //    (P * Psi)' * (P * Psi) = Psi' * (P'*P) * Psi = Psi' * Psi, and the result will
    //    still be the same as Psi'*Psi. P.S., however, this requires the domain partition
    //    to be UNIFORM!

    // The following code sets up blacscomm in the 1st way described above
    color = rank_dmcomm;
    if (pSPARC->bandcomm_index == -1 || pSPARC->dmcomm == MPI_COMM_NULL || pSPARC->kptcomm_index == -1)   color = INT_MAX;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    // split the kptcomm into several cblacscomms using color = rank_dmcomm
    color = (color >= 0) ? color : INT_MAX;
    MPI_Comm_split(pSPARC->kptcomm, color, pSPARC->bandcomm_index, &pSPARC->blacscomm);
#ifdef DEBUG
    t2 = MPI_Wtime();
    if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("\n--set up blacscomm took %.3f ms\n",(t2-t1)*1000);
#endif

#if defined(USE_MKL) || defined(USE_SCALAPACK)
    // copy this when need to create cblacs context using Cblacs_gridmap
    //int *usermap, ldumap;
    int size_blacscomm, DMnd, DMndsp, DMndspe;
    int *usermap, *usermap_0, *usermap_1;
    int info, bandsizes[2], nprow, npcol, myrow, mycol;

    size_blacscomm = pSPARC->npband;
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    DMndspe = DMnd * pSPARC->Nspinor_eig;

    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        usermap = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_0 = (int *)malloc(sizeof(int)*size_blacscomm);
        usermap_1 = (int *)malloc(sizeof(int)*size_blacscomm);
        for (i = 0; i < size_blacscomm; i++) {
            usermap[i] = usermap_0[i] = usermap_1[i] = rank0nuChi0EigscommInWorld + (i + rank - rank_kptcomm + rank_dmcomm * pSPARC->npband);
        }

        // in order to use a subgroup of blacscomm, use the following
        // to get a good number of subgroup processes
        bandsizes[0] = ((pSPARC->Nd-1)/pSPARC->npNd+1) * pSPARC->Nspinor_eig;
        bandsizes[1] = pSPARC->Nstates;
        //SPARC_Dims_create(pSPARC->npband, 2, bandsizes, 1, dims, &ierr);
        ScaLAPACK_Dims_2D_BLCYC(size_blacscomm, bandsizes, dims);
#ifdef DEBUG
        if ((!rank) && flagPrintNuChi0Eigscomm) printf("rank = %d, size_blacscomm = %d, ScaLAPACK topology Dims = (%d, %d)\n", rank, size_blacscomm, dims[0], dims[1]);
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
        usermap[0] = usermap_0[0] = usermap_1[0] = rank0nuChi0EigscommInWorld + rank;
        dims[0] = dims[1] = 1;
    }

#ifdef DEBUG
    if ((!rank) && flagPrintNuChi0Eigscomm) {
        printf("nproc = %d, size_blacscomm = %d = dims[0] * dims[1] = (%d, %d)\n", nproc, size_blacscomm, dims[0], dims[1]);
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 3) {
    //     printf("usermap of rank %d:", rank);
    //     for (i = 0; i < size_blacscomm; i++) {
    //         printf(" %d", usermap[i]);
    //     }
    //     printf("\n");
    // }
#endif
    // TODO: CHANGE USERMAP TO COLUMN MAJOR!
    int myrank_mpi, nprocs_mpi;
    Cblacs_pinfo( &myrank_mpi, &nprocs_mpi );

    // the following commands will create a context with handle ictxt_blacs
    Cblacs_get( -1, 0, &pSPARC->ictxt_blacs );
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        Cblacs_gridmap( &pSPARC->ictxt_blacs, usermap_0, 1, 1, pSPARC->npband); // row topology
    } else {
        Cblacs_gridmap( &pSPARC->ictxt_blacs, usermap_0, 1, 1, dims[0] * dims[1]); // row topology
    }
    free(usermap_0);

    // create ictxt_blacs topology
    Cblacs_get( -1, 0, &pSPARC->ictxt_blacs_topo );
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        // create usermap_1 = reshape(usermap,[dims[0], dims[1]])
        for (j = 0; j < dims[1]; j++) {
            for (i = 0; i < dims[0]; i++) {
                usermap_1[j*dims[0]+i] = usermap[i*dims[1]+j];
            }
        }
    }
    Cblacs_gridmap( &pSPARC->ictxt_blacs_topo, usermap_1, dims[0], dims[0], dims[1] ); // Cart topology
    free(usermap_1);
    free(usermap);

    // get coord of each process in original context
    Cblacs_gridinfo( pSPARC->ictxt_blacs, &nprow, &npcol, &myrow, &mycol );
    if ((!rank) && flagPrintNuChi0Eigscomm) printf("rank = %d, myrank_mpi %d, nprocs_mpi %d, bandcomm_index %d, dmcomm exist %d, myrow %d, nprow %d, mycol %d, npcol %d\n", rank, myrank_mpi, nprocs_mpi, pSPARC->bandcomm_index, (pSPARC->dmcomm != MPI_COMM_NULL), myrow, nprow, mycol, npcol);
    int ZERO = 0, mb, nb, llda;
    mb = max(1, DMndspe);
    nb = (pSPARC->Nstates - 1) / pSPARC->npband + 1; // equal to ceil(Nstates/npband), for int only
    // set up descriptor for storage of orbitals in ictxt_blacs (original)
    llda = max(1, DMndsp);
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(&pSPARC->desc_orbitals[0], &DMndspe, &pSPARC->Nstates,
                  &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs, &llda, &info);
    } else {
        for (i = 0; i < 9; i++)
            pSPARC->desc_orbitals[i] = 0;
    }
#ifdef DEBUG
    int temp_r, temp_c;
    temp_r = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
    temp_c = numroc_( &pSPARC->Nstates, &nb, &mycol, &ZERO, &npcol);
    if ((!rank) && flagPrintNuChi0Eigscomm) printf("rank = %2d, my blacs rank = %d, BLCYC size (%d, %d), actual size (%d, %d), DMndspe %d, mb %d, myrow %d, nprow %d, Nstates %d, nb %d, mycol %d, npcol %d\n", rank, pSPARC->bandcomm_index, temp_r, temp_c, DMndsp, pSPARC->Nband_bandcomm,
        DMndspe, mb, myrow, nprow, pSPARC->Nstates, nb, mycol, npcol);
#endif
    // get coord of each process in block cyclic topology context
    Cblacs_gridinfo( pSPARC->ictxt_blacs_topo, &nprow, &npcol, &myrow, &mycol );
    pSPARC->nprow_ictxt_blacs_topo = nprow;
    pSPARC->npcol_ictxt_blacs_topo = npcol;

    // set up descriptor for block-cyclic format storage of orbitals in ictxt_blacs
    // TODO: make block-cyclic parameters mb and nb input variables!
    mb = max(1, DMndspe / dims[0]); // this is only block, no cyclic! Tune this to improve efficiency!
    nb = max(1, pSPARC->Nstates / dims[1]); // this is only block, no cyclic!

    // find number of rows/cols of the local distributed orbitals
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        pSPARC->nr_orb_BLCYC = numroc_( &DMndspe, &mb, &myrow, &ZERO, &nprow);
        pSPARC->nc_orb_BLCYC = numroc_( &pSPARC->Nstates, &nb, &mycol, &ZERO, &npcol);
    } else {
        pSPARC->nr_orb_BLCYC = 1;
        pSPARC->nc_orb_BLCYC = 1;
    }
    llda = max(1, pSPARC->nr_orb_BLCYC);
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(&pSPARC->desc_orb_BLCYC[0], &DMndspe, &pSPARC->Nstates,
                  &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &llda, &info);
    } else {
        for (i = 0; i < 9; i++)
            pSPARC->desc_orb_BLCYC[i] = 0;
    }

    // set up distribution of projected Hamiltonian and the corresponding overlap matrix
    // TODO: Find optimal distribution of the projected Hamiltonian and mass matrix!
    //       For now Hp and Mp are distributed as follows: we distribute them in the same
    //       context topology as the bands.
    //       Note that mb = nb!

    // the maximum Nstates up to which we will use LAPACK to solve
    // the subspace eigenproblem in serial
    // int MAX_NS = 2000;
    int MAX_NS = pSPARC->eig_serial_maxns;
    pSPARC->useLAPACK = (pSPARC->Nstates <= MAX_NS) ? 1 : 0;

    int mbQ, nbQ, lldaQ;
    
    // block size for storing Hp and Mp
    if (pSPARC->useLAPACK == 1) {
        // in this case we will call LAPACK instead to solve the subspace eigenproblem
        mb = nb = pSPARC->Nstates;
        mbQ = nbQ = 64; // block size for storing subspace eigenvectors
    } else {
        // in this case we will use ScaLAPACK to solve the subspace eigenproblem
        mb = nb = pSPARC->eig_paral_blksz;
        mbQ = nbQ = pSPARC->eig_paral_blksz; // block size for storing subspace eigenvectors
    }
#ifdef DEBUG
    if ((!rank) && flagPrintNuChi0Eigscomm) printf("rank = %d, mb = nb = %d, mbQ = nbQ = %d\n", rank, mb, mbQ);
#endif
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        pSPARC->nr_Hp_BLCYC = pSPARC->nr_Mp_BLCYC = numroc_( &pSPARC->Nstates, &mb, &myrow, &ZERO, &nprow);
        pSPARC->nr_Hp_BLCYC = pSPARC->nr_Mp_BLCYC = max(1, pSPARC->nr_Mp_BLCYC);
        pSPARC->nc_Hp_BLCYC = pSPARC->nc_Mp_BLCYC = numroc_( &pSPARC->Nstates, &nb, &mycol, &ZERO, &npcol);
        pSPARC->nc_Hp_BLCYC = pSPARC->nc_Mp_BLCYC = max(1, pSPARC->nc_Mp_BLCYC);
        pSPARC->nr_Q_BLCYC = numroc_( &pSPARC->Nstates, &mbQ, &myrow, &ZERO, &nprow);
        pSPARC->nc_Q_BLCYC = numroc_( &pSPARC->Nstates, &nbQ, &mycol, &ZERO, &npcol);
    } else {
        pSPARC->nr_Hp_BLCYC = pSPARC->nc_Hp_BLCYC = 1;
        pSPARC->nr_Mp_BLCYC = pSPARC->nc_Mp_BLCYC = 1;
        pSPARC->nr_Q_BLCYC  = pSPARC->nc_Q_BLCYC  = 1;
    }

    llda = max(1, pSPARC->nr_Hp_BLCYC);
    lldaQ= max(1, pSPARC->nr_Q_BLCYC);
    if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
        descinit_(&pSPARC->desc_Hp_BLCYC[0], &pSPARC->Nstates, &pSPARC->Nstates,
                  &mb, &nb, &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &llda, &info);
        for (i = 0; i < 9; i++) {
            //pSPARC->desc_Q_BLCYC[i] = pSPARC->desc_Mp_BLCYC[i] = pSPARC->desc_Hp_BLCYC[i];
            pSPARC->desc_Mp_BLCYC[i] = pSPARC->desc_Hp_BLCYC[i];
        }
        descinit_(&pSPARC->desc_Q_BLCYC[0], &pSPARC->Nstates, &pSPARC->Nstates,
                  &mbQ, &nbQ, &ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &lldaQ, &info);
    } else {
        for (i = 0; i < 9; i++) {
            pSPARC->desc_Q_BLCYC[i] = pSPARC->desc_Mp_BLCYC[i] = pSPARC->desc_Hp_BLCYC[i] = 0;
        }
    }

#ifdef DEBUG
    if (!rank) printf("rank = %d, nr_Hp = %d, nc_Hp = %d\n", rank, pSPARC->nr_Hp_BLCYC, pSPARC->nc_Hp_BLCYC);
#endif

    // allocate memory for block cyclic distribution of projected Hamiltonian and mass matrix
    if (pSPARC->isGammaPoint){
        pSPARC->Hp = (double *)malloc(pSPARC->nr_Hp_BLCYC * pSPARC->nc_Hp_BLCYC * sizeof(double));
        pSPARC->Mp = (double *)malloc(pSPARC->nr_Mp_BLCYC * pSPARC->nc_Mp_BLCYC * sizeof(double));
        pSPARC->Q  = (double *)malloc(pSPARC->nr_Q_BLCYC * pSPARC->nc_Q_BLCYC * sizeof(double));
    } else{
        pSPARC->Hp_kpt = (double _Complex *) malloc(pSPARC->nr_Hp_BLCYC * pSPARC->nc_Hp_BLCYC * sizeof(double _Complex));
        pSPARC->Mp_kpt = (double _Complex *) malloc(pSPARC->nr_Mp_BLCYC * pSPARC->nc_Mp_BLCYC * sizeof(double _Complex));
        pSPARC->Q_kpt  = (double _Complex *) malloc(pSPARC->nr_Q_BLCYC * pSPARC->nc_Q_BLCYC * sizeof(double _Complex));
    }
#else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    pSPARC->useLAPACK = 1;
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    //------------------------------------------------//
    //               set up poisson domain            //
    //------------------------------------------------//
    // some variables are reused here
    npNd = pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi;
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    minsize = pSPARC->order/2;

    if (npNd == 0)  {
        // when user does not provide domain decomposition parameters
        npNd = nproc; // try to use all processors
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution."
                   "Please check if your gridsizes are less than MINSIZE\n");
        }
        pSPARC->npNdx_phi = dims[0];
        pSPARC->npNdy_phi = dims[1];
        pSPARC->npNdz_phi = dims[2];
    } else if (npNd < 0 || npNd > nproc || pSPARC->Nx / pSPARC->npNdx_phi < minsize ||
               pSPARC->Ny / pSPARC->npNdy_phi < minsize || pSPARC->Nz / pSPARC->npNdz_phi < minsize) {
        // when domain decomposition parameters are not reasonable
        npNd = nproc;
        SPARC_Dims_create(npNd, 3, gridsizes, minsize, dims, &ierr);
        if (ierr == 1 && rank == 0) {
            printf("WARNING: error occured when calculating best domain distribution."
                   "Please check if your gridsizes are less than MINSIZE\n");
        }
        pSPARC->npNdx_phi = dims[0];
        pSPARC->npNdy_phi = dims[1];
        pSPARC->npNdz_phi = dims[2];
    } else {
        dims[0] = pSPARC->npNdx_phi;
        dims[1] = pSPARC->npNdy_phi;
        dims[2] = pSPARC->npNdz_phi;
    }

    // recalculate number of processors in dmcomm_phi
    npNd = pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi;

    periods[0] = 1 - pSPARC->BCx;
    periods[1] = 1 - pSPARC->BCy;
    periods[2] = 1 - pSPARC->BCz;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // Processes in bandcomm with rank >= dims[0]*dims[1]*…*dims[d-1] will return MPI_COMM_NULL
    MPI_Cart_create(nuChi0Eigscomm, 3, dims, periods, 1, &pSPARC->dmcomm_phi); // 1 is to reorder rank

#ifdef DEBUG
    if ((rank == 0) && flagPrintNuChi0Eigscomm) {
        printf("========================================================================\n"
                   "Poisson domain decomposition:"
                   "np total = %d, {Nx, Ny, Nz} = {%d, %d, %d}\n"
                   "nproc used = %d = {%d, %d, %d}, nodes/proc = {%.2f, %.2f, %.2f}\n\n",
                   nproc,pSPARC->Nx,pSPARC->Ny,pSPARC->Nz,dims[0]*dims[1]*dims[2],dims[0],dims[1],dims[2],pSPARC->Nx/(double)dims[0],pSPARC->Ny/(double)dims[1],pSPARC->Nz/(double)dims[2]);
    }
#endif

    // find the vertices of the domain in each processor
    // pSPARC->DMVertices[6] = [xs,xe,ys,ye,zs,ze]
    // int Nx_dist, Ny_dist, Nz_dist;
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        MPI_Comm_rank(pSPARC->dmcomm_phi, &rank_dmcomm);
        MPI_Cart_coords(pSPARC->dmcomm_phi, rank_dmcomm, 3, coord_dmcomm);

        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;

        // find size of distributed domain
        pSPARC->Nx_d = block_decompose(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->Ny_d = block_decompose(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->Nz_d = block_decompose(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->Nd_d = pSPARC->Nx_d * pSPARC->Ny_d * pSPARC->Nz_d;

        // find corners of the distributed domain
        pSPARC->DMVertices[0] = block_decompose_nstart(gridsizes[0], dims[0], coord_dmcomm[0]);
        pSPARC->DMVertices[1] = pSPARC->DMVertices[0] + pSPARC->Nx_d - 1;
        pSPARC->DMVertices[2] = block_decompose_nstart(gridsizes[1], dims[1], coord_dmcomm[1]);
        pSPARC->DMVertices[3] = pSPARC->DMVertices[2] + pSPARC->Ny_d - 1;
        pSPARC->DMVertices[4] = block_decompose_nstart(gridsizes[2], dims[2], coord_dmcomm[2]);
        pSPARC->DMVertices[5] = pSPARC->DMVertices[4] + pSPARC->Nz_d - 1;
    } else {
        rank_dmcomm = -1;
        coord_dmcomm[0] = -1; coord_dmcomm[1] = -1; coord_dmcomm[2] = -1;
        pSPARC->Nx_d = 0;
        pSPARC->Ny_d = 0;
        pSPARC->Nz_d = 0;
        pSPARC->Nd_d = 0;
        pSPARC->DMVertices[0] = 0;
        pSPARC->DMVertices[1] = 0;
        pSPARC->DMVertices[2] = 0;
        pSPARC->DMVertices[3] = 0;
        pSPARC->DMVertices[4] = 0;
        pSPARC->DMVertices[5] = 0;
    }

    // Set up a 26 neighbor communicator for nonorthogonal systems
    // TODO: Modify the communicator based on number of non zero enteries in off diagonal of pSPARC->lapcT
    // TODO: Take into account more than 26 Neighbors
    if(pSPARC->cell_typ != 0) {
        if(pSPARC->dmcomm_phi != MPI_COMM_NULL) {
            int rank_chk, dir, tmp;
            int ncoords[3];
            int nnproc = 1; // No. of neighboring processors in each direction
            int nneighb = pow(2*nnproc+1,3) - 1; // 6 faces + 12 edges + 8 corners
            int *neighb;
            neighb = (int *) malloc(nneighb * sizeof (int));
            int count = 0, i, j, k;
            for(k = -nnproc; k <= nnproc; k++){
                for(j = -nnproc; j <= nnproc; j++){
                    for(i = -nnproc; i <= nnproc; i++){
                        tmp = 0;
                        if(i == 0 && j == 0 && k == 0){
                            continue;
                        } else{
                            ncoords[0] = coord_dmcomm[0] + i;
                            ncoords[1] = coord_dmcomm[1] + j;
                            ncoords[2] = coord_dmcomm[2] + k;
                            for(dir = 0; dir < 3; dir++){
                                if(periods[dir]){
                                    if(ncoords[dir] < 0)
                                        ncoords[dir] += dims[dir];
                                    else if(ncoords[dir] >= dims[dir])
                                        ncoords[dir] -= dims[dir];
                                    tmp = 1;
                                } else{
                                    if(ncoords[dir] < 0 || ncoords[dir] >= dims[dir]){
                                        rank_chk = MPI_PROC_NULL;
                                        tmp = 0;
                                        break;
                                    }
                                    else
                                        tmp = 1;
                                }
                            }
                            //TODO: For dirchlet give rank = MPI_PROC_NULL for out of bounds coordinates
                            if(tmp == 1)
                                MPI_Cart_rank(pSPARC->dmcomm_phi,ncoords,&rank_chk); // proc rank corresponding to ncoords_mapped

                            neighb[count] = rank_chk;
                            count++;
                        }
                    }
                }
            }
            MPI_Dist_graph_create_adjacent(pSPARC->dmcomm_phi,nneighb,neighb,(int *)MPI_UNWEIGHTED,nneighb,neighb,(int *)MPI_UNWEIGHTED,MPI_INFO_NULL,0,&pSPARC->comm_dist_graph_phi); // creates a distributed graph topology (adjacent, cartesian cubical)
            //pSPARC->dmcomm_phi = pSPARC->comm_dist_graph_phi;
            free(neighb);
        }
    }

#ifdef DEBUG
    t2 = MPI_Wtime();
    if ((rank == 0) && flagPrintNuChi0Eigscomm) printf("\n--set up dmcomm_phi took %.3f ms\n",(t2-t1)*1000);
#endif

    // allocate memory for storing eigenvalues
    pSPARC->lambda = (double *)calloc(pSPARC->Nstates * pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof(double));
    assert(pSPARC->lambda != NULL);

    pSPARC->lambda_sorted = pSPARC->lambda;

    // allocate memory for storing eigenvalues
    pSPARC->occ = (double *)calloc(pSPARC->Nstates * pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof(double));
    assert(pSPARC->occ != NULL);

    pSPARC->occ_sorted = pSPARC->occ;

    pSPARC->eigmin = (double *) calloc(pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof (double));
    pSPARC->eigmax = (double *) calloc(pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm, sizeof (double));

    /* allocate memory for storing atomic forces*/
    pSPARC->forces = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
    assert(pSPARC->forces != NULL);

    if (pSPARC->dmcomm != MPI_COMM_NULL && pSPARC->bandcomm_index >= 0) {
        pSPARC->Veff_loc_dmcomm = (double *)malloc( pSPARC->Nd_d_dmcomm * pSPARC->Nspden * sizeof(double) );
        assert(pSPARC->Veff_loc_dmcomm != NULL);
    }

    pSPARC->Veff_loc_kptcomm_topo = (double *)malloc( pSPARC->Nd_d_kptcomm * ((pSPARC->spin_typ == 2) ? 4 : 1) *  sizeof(double) );
    assert(pSPARC->Veff_loc_kptcomm_topo != NULL);

    // allocate memory for initial guess vector for Lanczos
    if (pSPARC->isGammaPoint && pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        pSPARC->Lanczos_x0 = (double *)malloc(pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_eig * sizeof(double));
        assert(pSPARC->Lanczos_x0 != NULL);
    }

    if (pSPARC->isGammaPoint != 1 && pSPARC->kptcomm_topo != MPI_COMM_NULL) {
        pSPARC->Lanczos_x0_complex = (double _Complex *)malloc(pSPARC->Nd_d_kptcomm * pSPARC->Nspinor_eig * sizeof(double _Complex));
        assert(pSPARC->Lanczos_x0_complex != NULL);
    }

    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        /* allocate memory for electrostatics calculation */
        int DMnx = pSPARC->DMVertices[1] - pSPARC->DMVertices[0] + 1;
        int DMny = pSPARC->DMVertices[3] - pSPARC->DMVertices[2] + 1;
        int DMnz = pSPARC->DMVertices[5] - pSPARC->DMVertices[4] + 1;
        int DMnd = DMnx * DMny * DMnz;
        // allocate memory for electron density (sum of atom potential) and charge density
        pSPARC->electronDens_at = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->electronDens_core = (double *)calloc( DMnd, sizeof(double) );
        pSPARC->psdChrgDens = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->psdChrgDens_ref = (double *)malloc( DMnd * sizeof(double) );
        pSPARC->Vc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->electronDens_core != NULL);
        assert(pSPARC->electronDens_at != NULL && pSPARC->psdChrgDens != NULL &&
               pSPARC->psdChrgDens_ref != NULL && pSPARC->Vc != NULL);
        // allocate memory for electron density
        pSPARC->electronDens = (double *)malloc( DMnd * pSPARC->Nspdentd * sizeof(double) );
        assert(pSPARC->electronDens != NULL);
        // allocate memory for magnetization
        if (pSPARC->spin_typ > 0) {
            pSPARC->mag = (double *)malloc( DMnd * pSPARC->Nmag * sizeof(double) );
            assert(pSPARC->mag != NULL);
            int ncol = (pSPARC->spin_typ > 1) + pSPARC->spin_typ; // 0 0 1 3
            pSPARC->mag_at = (double *)malloc( DMnd * ncol * sizeof(double) );
            assert(pSPARC->mag_at != NULL);
            pSPARC->AtomMag = (double *)malloc( (pSPARC->spin_typ == 2 ? 3 : 1) * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->AtomMag != NULL);
        }
        // allocate memory for charge extrapolation arrays
        if(pSPARC->MDFlag == 1 || pSPARC->RelaxFlag == 1 || pSPARC->RelaxFlag == 3){
            pSPARC->delectronDens = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens != NULL);
            pSPARC->delectronDens_0dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_0dt != NULL);
            pSPARC->delectronDens_1dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_1dt != NULL);
            pSPARC->delectronDens_2dt = (double *)malloc( DMnd * sizeof(double) );
            assert(pSPARC->delectronDens_2dt != NULL);
            pSPARC->atom_pos_nm = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_nm != NULL);
            pSPARC->atom_pos_0dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_0dt != NULL);
            pSPARC->atom_pos_1dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_1dt != NULL);
            pSPARC->atom_pos_2dt = (double *)malloc( 3 * pSPARC->n_atom * sizeof(double) );
            assert(pSPARC->atom_pos_2dt != NULL);
        }
        // allocate memory for electrostatic potential
        pSPARC->elecstPotential = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->elecstPotential != NULL);

        // allocate memory for XC potential
        pSPARC->XCPotential = (double *)malloc( DMnd * pSPARC->Nspdend * sizeof(double) );
        assert(pSPARC->XCPotential != NULL);

        // allocate memory for exchange-correlation energy density
        pSPARC->e_xc = (double *)malloc( DMnd * sizeof(double) );
        assert(pSPARC->e_xc != NULL);

        // if GGA then allocate for xc energy per particle for each grid point and der. wrt. grho
        if(pSPARC->isgradient) {
            pSPARC->Dxcdgrho = (double *)malloc( DMnd * pSPARC->Nspdentd * sizeof(double) );
            assert(pSPARC->Dxcdgrho != NULL);
        }

        pSPARC->Veff_loc_dmcomm_phi = (double *)malloc(DMnd * pSPARC->Nspden * sizeof(double));        
        pSPARC->mixing_hist_xk      = (double *)malloc(DMnd * pSPARC->Nspden * sizeof(double));
        pSPARC->mixing_hist_fk      = (double *)calloc(DMnd * pSPARC->Nspden , sizeof(double));
        pSPARC->mixing_hist_fkm1    = (double *)calloc(DMnd * pSPARC->Nspden , sizeof(double));
        pSPARC->mixing_hist_xkm1    = (double *)malloc(DMnd * pSPARC->Nspden * sizeof(double));
        pSPARC->mixing_hist_Xk      = (double *)malloc(DMnd * pSPARC->Nspden * pSPARC->MixingHistory * sizeof(double));
        pSPARC->mixing_hist_Fk      = (double *)malloc(DMnd * pSPARC->Nspden * pSPARC->MixingHistory * sizeof(double));
        pSPARC->mixing_hist_Pfk     = (double *)calloc(DMnd * pSPARC->Nspden, sizeof(double));
        assert(pSPARC->Veff_loc_dmcomm_phi != NULL && pSPARC->mixing_hist_xk   != NULL &&
               pSPARC->mixing_hist_fk      != NULL && pSPARC->mixing_hist_fkm1 != NULL &&
               pSPARC->mixing_hist_xkm1    != NULL && pSPARC->mixing_hist_Xk   != NULL &&
               pSPARC->mixing_hist_Fk      != NULL && pSPARC->mixing_hist_Pfk  != NULL);

        if (pSPARC->MixingVariable == 1) {
            pSPARC->Veff_loc_dmcomm_phi_in = (double *)malloc(DMnd * pSPARC->Nspdend * sizeof(double));
            assert(pSPARC->Veff_loc_dmcomm_phi_in != NULL);
        } 

        if (pSPARC->MixingVariable == 0 && pSPARC->spin_typ) {
            pSPARC->electronDens_in = (double *)malloc(DMnd * pSPARC->Nspdentd * sizeof(double));
            assert(pSPARC->electronDens_in != NULL);
        }

        // The following rho_in and phi_in are only used for evaluating QE scf errors
        if (pSPARC->scf_err_type == 1) {
            pSPARC->rho_dmcomm_phi_in = (double *)malloc(DMnd * sizeof(double));
            assert(pSPARC->rho_dmcomm_phi_in != NULL);
            pSPARC->phi_dmcomm_phi_in = (double *)malloc(DMnd * sizeof(double));
            assert(pSPARC->phi_dmcomm_phi_in != NULL);
        }

        // initialize electrostatic potential as random guess vector
        if (pSPARC->FixRandSeed == 1) {
            SeededRandVec(pSPARC->elecstPotential, pSPARC->DMVertices, gridsizes, -1.0, 1.0, 0);
        } else {
            srand(rank+1);
            double rand_min = -1.0, rand_max = 1.0;
            for (i = 0; i < DMnd; i++) {
                pSPARC->elecstPotential[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX; // or 1.0
            }
        }

        if (pSPARC->spin_typ == 2) {
            // allocate memory for XC potential
            pSPARC->XCPotential_nc = (double *)malloc( DMnd * pSPARC->Nspden * sizeof(double) );
            assert(pSPARC->XCPotential_nc != NULL);
            if (pSPARC->MixingVariable == 1) {
                pSPARC->Veff_dia_loc_dmcomm_phi = (double *)malloc(DMnd * pSPARC->Nspdend * sizeof(double));
                assert(pSPARC->Veff_dia_loc_dmcomm_phi != NULL);
            } 
        }

        if (pSPARC->BC == 1) {
            pSPARC->MpExp = (MPEXP_OBJ *) malloc(sizeof(MPEXP_OBJ));
            init_multipole_expansion(pSPARC, pSPARC->MpExp, 
                pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, pSPARC->Nx_d, pSPARC->Ny_d, pSPARC->Nz_d, pSPARC->DMVertices, pSPARC->dmcomm_phi);
        }
    }

    // Set up D2D target objects between phi comm and psi comm
    // Note: Set_D2D_target require gridsizes, rdims, sdims to be global in union{send_comm, recv_comm},
    //       i.e., all processes know about these values and they are the same in all processes
    int sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx;
    gridsizes[1] = pSPARC->Ny;
    gridsizes[2] = pSPARC->Nz;
    rdims[0] = pSPARC->npNdx;
    rdims[1] = pSPARC->npNdy;
    rdims[2] = pSPARC->npNdz;
    sdims[0] = pSPARC->npNdx_phi;
    sdims[1] = pSPARC->npNdy_phi;
    sdims[2] = pSPARC->npNdz_phi;

    Set_D2D_Target(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, gridsizes, pSPARC->DMVertices, pSPARC->DMVertices_dmcomm, pSPARC->dmcomm_phi,
                   sdims, (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, rdims, nuChi0Eigscomm);

    // Set up D2D target objects between psi comm and kptcomm_topo comm
    // check if kptcomm_topo is the same as dmcomm_phi
    // If found rank order in the cartesian topology is different for dmcomm_phi and 
    // kptcomm_topo, consider to add MPI_Bcast for is_phi_eq_kpt_topo
    if ((pSPARC->npNdx_phi == pSPARC->npNdx_kptcomm) && 
        (pSPARC->npNdy_phi == pSPARC->npNdy_kptcomm) && 
        (pSPARC->npNdz_phi == pSPARC->npNdz_kptcomm))
        pSPARC->is_phi_eq_kpt_topo = 1;
    else
        pSPARC->is_phi_eq_kpt_topo = 0;

    if (((pSPARC->chefsibound_flag == 0 || pSPARC->chefsibound_flag == 1) &&
            pSPARC->spincomm_index >=0 && pSPARC->kptcomm_index >= 0
            && (pSPARC->spin_typ != 0 || !pSPARC->is_phi_eq_kpt_topo || !pSPARC->isGammaPoint))
            || (pSPARC->usefock > 0) )
    {
        gridsizes[0] = pSPARC->Nx;
        gridsizes[1] = pSPARC->Ny;
        gridsizes[2] = pSPARC->Nz;
        sdims[0] = pSPARC->npNdx;
        sdims[1] = pSPARC->npNdy;
        sdims[2] = pSPARC->npNdz;
        rdims[0] = pSPARC->npNdx_kptcomm;
        rdims[1] = pSPARC->npNdy_kptcomm;
        rdims[2] = pSPARC->npNdz_kptcomm;

        Set_D2D_Target(&pSPARC->d2d_dmcomm_lanczos, &pSPARC->d2d_kptcomm_topo, gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->DMVertices_kptcomm,
                       pSPARC->bandcomm_index == 0 ? pSPARC->dmcomm : MPI_COMM_NULL, sdims,
                       pSPARC->kptcomm_topo, rdims, pSPARC->kptcomm);
    }

    // parallelization summary
    #ifdef DEBUG
    if ((rank == 0) && flagPrintNuChi0Eigscomm) {
        printf("\n");
        printf("-----------------------------------------------\n");
        printf("Parallelization summary\n");
        printf("Total number of processors: %d\n", nproc);
        printf("-----------------------------------------------\n");
        printf("== Psi domain ==\n");
        printf("Total number of processors used for Psi domain: %d\n", pSPARC->npspin*pSPARC->npkpt*pSPARC->npband*pSPARC->npNd);
        printf("npspin  : %d\n", pSPARC->npspin);
        printf("# of spin per spincomm           : %.0f\n", ceil(pSPARC->Nspin / (double)pSPARC->npspin));
        printf("npkpt   : %d\n", pSPARC->npkpt);
        printf("# of k-points per kptcomm        : %.0f\n", ceil(pSPARC->Nkpts_sym / (double)pSPARC->npkpt));
        printf("npband  : %d\n", pSPARC->npband);
        printf("# of bands per bandcomm          : %.0f\n", ceil(pSPARC->Nstates / (double)pSPARC->npband));
        printf("npdomain: %d\n", pSPARC->npNd);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx, pSPARC->npNdy, pSPARC->npNdz);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSPARC->Nd_d_dmcomm,pSPARC->Nx_d_dmcomm,pSPARC->Ny_d_dmcomm,pSPARC->Nz_d_dmcomm);
        printf("-----------------------------------------------\n");
        printf("== Phi domain ==\n");
        printf("Total number of processors used for Phi domain: %d\n", pSPARC->npNdx_phi * pSPARC->npNdy_phi * pSPARC->npNdz_phi);
        printf("Embeded Cartesian topology dims: (%d,%d,%d)\n", pSPARC->npNdx_phi,pSPARC->npNdy_phi, pSPARC->npNdz_phi);
        printf("# of FD-grid points per processor: %d = (%d,%d,%d)\n", pSPARC->Nd_d,pSPARC->Nx_d,pSPARC->Ny_d,pSPARC->Nz_d);
        printf("-----------------------------------------------\n");
    }
    #endif
}


