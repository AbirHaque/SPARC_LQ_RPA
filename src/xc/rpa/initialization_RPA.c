/**
 * @file    initialization.c
 * @brief   This file contains the initializing function for RPA calculation
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
#include <time.h>
#include <assert.h>
// this is for checking existence of files
# include <unistd.h>

#include "initialization.h"
#include "tools.h"
#include "readfiles.h"
#include "kroneckerLaplacian.h"
#include "exactExchangeInitialization.h"
#include "electrostatics.h"

#include "initialization_RPA.h"
#include "initialization_SPARC.h"
#include "parallelization_RPA.h"
#include "parallelization_SPARC.h"
#include "generateKgrid.h"

#define N_MEMBR_RPA 30

void initialize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int argc, char* argv[]) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t1;
    if (!rank) {
        t1 = MPI_Wtime(); 
        pSPARC->time_start = t1;
    }
    Initialize_SPARC_before_SetComm(pSPARC, argc, argv); // include cell size, lattice vectors, mesh size and k-point grid, reading ion file & pseudopotentials
    pRPA->deltaRhos = NULL;
    pRPA->deltaVs = NULL;
    pRPA->Ys = NULL;
    pRPA->sprtNuDeltaVs = NULL;
    pRPA->deltaPsisReal = NULL;
    pRPA->deltaPsisImag = NULL;
    pRPA->allXorb = NULL;
    pRPA->allLambdas = NULL;
    
    pRPA->deltaRhos_kpt = NULL;
    pRPA->deltaVs_kpt = NULL;
    pRPA->Ys_kpt = NULL;
    pRPA->sprtNuDeltaVs_kpt = NULL;
    pRPA->deltaPsis_kpt = NULL;
    pRPA->allXorb_kpt = NULL;
    pRPA->Nkpts_sym = pSPARC->Nkpts_sym;
    pRPA->kptWts = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    pRPA->k1 = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    pRPA->k2 = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    pRPA->k3 = (double *)malloc(pRPA->Nkpts_sym * sizeof(double));
    transfer_kpoints(pSPARC, pRPA); // pRPA saves symmetric k-point
    recalculate_kpoints(pSPARC); // pSPARC saves k-point complete grid, no symmetry. pSPARC->Nkpts_sym = pSPARC->Nkpts
    RPA_INPUT_OBJ RPA_Input;
    /* Create new MPI struct datatype RPA_INPUT_MPI (for broadcasting) */
    MPI_Datatype RPA_INPUT_MPI;
    RPA_Input_MPI_create(&RPA_INPUT_MPI);
    if (!rank) {
        set_RPA_defaults(&RPA_Input, pSPARC->Nstates, pSPARC->Nd);
        strncpy(RPA_Input.filename, pSPARC->filename, L_STRING);
        strncpy(RPA_Input.filename_out, pSPARC->OutFilename, L_STRING);
        read_RPA_inputs(&RPA_Input); 
        printf("read block size %d\n", RPA_Input.SternBlockSize[7]);
        MPI_Bcast(&RPA_Input, 1, RPA_INPUT_MPI, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&RPA_Input, 1, RPA_INPUT_MPI, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&RPA_INPUT_MPI); // free the new MPI datatype
    RPA_copy_inputs(pRPA,&RPA_Input);
    if(pRPA->flagLQ){
        pRPA->nuChi0Neig=pSPARC->Nx*pSPARC->Ny*pSPARC->Nz;
    }
    // set q-point grid
    pRPA->Nqpts_sym = pSPARC->Kx*pSPARC->Ky*pSPARC->Kz,
    pRPA->qptWts = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->q1 = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->q2 = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->q3 = (double *)malloc(pRPA->Nqpts_sym * sizeof(double));
    pRPA->Nqpts_sym = set_qpoints(pRPA->qptWts, pRPA->q1, pRPA->q2, pRPA->q3, pSPARC->Kx, pSPARC->Ky, pSPARC->Kz, pSPARC->range_x, pSPARC->range_y, pSPARC->range_z);
    pRPA->kPqSymList = (int **)malloc(pSPARC->Nkpts_sym * sizeof(int*)); // pSPARC saves k-point complete grid, no symmetry. pSPARC->Nkpts_sym = pSPARC->Nkpts
    pRPA->kPqList = (int**)malloc(pSPARC->Nkpts_sym * sizeof(int*));
    pRPA->kMqList = (int**)malloc(pSPARC->Nkpts_sym * sizeof(int*));
    for (int nk = 0; nk < pSPARC->Nkpts_sym; nk++) {
        pRPA->kPqSymList[nk] = (int*)malloc((pRPA->Nqpts_sym + 1) * sizeof(int));
        pRPA->kPqList[nk] = (int*)malloc(pRPA->Nqpts_sym * sizeof(int));
        pRPA->kMqList[nk] = (int*)malloc(pRPA->Nqpts_sym * sizeof(int));
    }
    set_kPq_kMq_lists(pRPA->Nkpts_sym, pRPA->k1, pRPA->k2, pRPA->k3, pSPARC->Nkpts_sym, pSPARC->k1, pSPARC->k2, pSPARC->k3, 
        pRPA->Nqpts_sym, pRPA->q1, pRPA->q2, pRPA->q3, pSPARC->range_x, pSPARC->range_y, pSPARC->range_z, pRPA->kPqSymList, pRPA->kPqList, pRPA->kMqList);
    pSPARC->kron_lap_exx = (KRON_LAP **) malloc(sizeof(KRON_LAP*) * pRPA->Nqpts_sym);
    // set integration point omegas
    pRPA->omega = (double *)malloc(pRPA->Nomega * sizeof(double));
    pRPA->omega01 = (double *)malloc(pRPA->Nomega * sizeof(double));
    pRPA->omegaWts = (double *)malloc(pRPA->Nomega * sizeof(double));
    set_omegas(pRPA->omega, pRPA->omega01, pRPA->omegaWts, pRPA->Nomega);
    pRPA->RRnuChi0Eigs = (double *)malloc(pRPA->nuChi0Neig * sizeof(double));
    if(!pRPA->flagLQ){
        pRPA->RRnuChi0EigVecs = (double *)malloc(pRPA->nuChi0Neig * pRPA->nuChi0Neig * sizeof(double));
        assert(pRPA->RRnuChi0EigVecs);
    }
    pRPA->ErpaTerms = (double *)malloc(pRPA->Nqpts_sym * pRPA->Nomega * sizeof(double));
    pRPA->eig_paral_maxnp = -1;
    pRPA->flagAppliedPrecond = 0;
    // for structure RPA, allocating space for delta orbitals, delta density, delta \nu\chi\Delta V...
    // printf("%s\n", pRPA->filename_out);
    Setup_Comms_RPA(pRPA, pSPARC->Nspin, pSPARC->Nkpts, pSPARC->Nstates, pSPARC->Nd); // set communicator for RPA calculation
    pSPARC->npspin = pRPA->npspin;
    pSPARC->npkpt = pRPA->npkpt;
    pSPARC->npband = pRPA->npband;
    pSPARC->npNdx = 1; // currently there is no domain parallelization
    pSPARC->npNdy = 1;
    pSPARC->npNdz = 1;
    pSPARC->npNdx_phi = 0;
    pSPARC->npNdy_phi = 0;
    pSPARC->npNdz_phi = 0;
    Initialize_SPARC_SetComm_after(pSPARC, pRPA->nuChi0Eigscomm, pRPA->nuChi0EigscommIndex, pRPA->rank0nuChi0EigscommInWorld);
    pRPA->npspin = pSPARC->npspin;
    pRPA->npkpt = pSPARC->npkpt;
    pRPA->npband = pSPARC->npband;
    snprintf(pRPA->timeRecordname, L_STRING, "timingRecords_rank%d_%dnuChi0Eigscomm.log", rank, pRPA->nuChi0EigscommIndex);
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    setup_blacsComm_RPA(pRPA, flagNoDmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_spincomm, pSPARC->Nspinor_eig, 
        pSPARC->Nd, pSPARC->eig_serial_maxns, pSPARC->eig_paral_blksz, pSPARC->isGammaPoint, 1); // all processors should get into this function!
    MPI_Barrier(MPI_COMM_WORLD);
    if (pRPA->npnuChi0Neig < 2) {
        pRPA->flagCyclicPermute = 0;
    }
    t1 = MPI_Wtime();
    if (pRPA->flagCyclicPermute && pRPA->nuChi0EigscommIndex != -1 && !flagNoDmcomm && pRPA->npnuChi0Neig != 1) {
        // if (pRPA->flagSQ) {
        //     generate_permute_matrix_SQ(pRPA->desc_permute_BLCYC, pRPA->rPermute, pRPA->nr_Q_BLCYC, pRPA->nc_Q_BLCYC, 
        //         pRPA->permute_BLCYC, pRPA->desc_Q_BLCYC, pRPA->Q, pRPA->ictxt_blacs, pRPA->nuChi0BlacsComm);
        // } else {
        if (pSPARC->isGammaPoint) {
            generate_permute_matrix(pRPA->desc_permute, pRPA->desc_permute_BLCYC, pRPA->rPermute, pRPA->cPermute,
                pRPA->permute, pRPA->permute_BLCYC, pRPA->ictxt_blacs, pRPA->nuChi0BlacsComm);
        } else {
            generate_permute_matrix_kpt(pRPA->desc_permute, pRPA->desc_permute_BLCYC, pRPA->rPermute, pRPA->cPermute,
                pRPA->permute_kpt, pRPA->permute_BLCYC_kpt, pRPA->ictxt_blacs, pRPA->nuChi0BlacsComm);
        }
        // }
    }
    MPI_Barrier(MPI_COMM_WORLD); // code above are okay
    double t2 = MPI_Wtime();
    if (!rank) printf("the time for composing permute matrix is %.3f ms\n", (t2 - t1)*1e3);
    if (pRPA->nuChi0EigscommIndex != -1) {
        if (pSPARC->isGammaPoint) {
            pRPA->deltaVs = (double*)calloc(sizeof(double), pSPARC->Nd * pRPA->nNuChi0Eigscomm); // to be modified when there is domain decomposition.
            assert(pRPA->deltaVs);
            pRPA->Ys = (double*)calloc(sizeof(double), pSPARC->Nd * pRPA->nNuChi0Eigscomm);
            assert(pRPA->Ys);
            // the reason for setting it out of flagNoDmcomm is for broadcasting deltaVs over nuChi0Eigscomm. Also, Nd_d_dmcomm is 0 if dmcomm is NULL.
        } else {
            pRPA->deltaVs_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd * pRPA->nNuChi0Eigscomm);
            assert(pRPA->deltaVs_kpt);
            pRPA->Ys_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd * pRPA->nNuChi0Eigscomm);
            assert(pRPA->Ys_kpt);
        }
    }
    if ((pRPA->nuChi0EigscommIndex != -1) && (!flagNoDmcomm)) {
        if (pSPARC->isGammaPoint) {
            pRPA->nearbyBandIndicesGamma = (int*)calloc(sizeof(int), pSPARC->Nspin_spincomm * 2 * pSPARC->Nband_bandcomm);
            pRPA->neighborBandIndicesGamma = NULL;
            pRPA->neighborBandsGamma = NULL;
            pRPA->allEpsilonsGamma = (double*)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nstates);
            pRPA->deltaRhos = (double*)calloc(sizeof(double), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            assert(pRPA->deltaRhos);
            pRPA->sprtNuDeltaVs = (double*)calloc(sizeof(double), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            assert(pRPA->sprtNuDeltaVs);
            pRPA->deltaPsisReal = (double*)calloc(sizeof(double), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            assert(pRPA->deltaPsisReal);
            pRPA->deltaPsisImag = (double*)calloc(sizeof(double), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            assert(pRPA->deltaPsisImag);
            if (pRPA->flagCOCGinitial) {
                pRPA->allXorb = (double*)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nd_d_dmcomm * pSPARC->Nstates);
                assert(pRPA->allXorb);
                pRPA->allLambdas = (double*)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nstates);
            }

            pSPARC->kron_lap_exx[0] = (KRON_LAP *) malloc(sizeof(KRON_LAP));
            init_kron_Lap(pSPARC, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, 
                pSPARC->BCx, pSPARC->BCy, pSPARC->BCz, pRPA->q1[0], pRPA->q2[0], pRPA->q3[0], pSPARC->isGammaPoint, pSPARC->kron_lap_exx[0]);
            if (pSPARC->BC == 1) {
                KRON_LAP* kron_lap = pSPARC->kron_lap_exx[0];
                // dirichlet BC needs multipole expansion
                // int DMVertices[6] = {0, pSPARC->Nx-1, 0, pSPARC->Ny-1, 0, pSPARC->Nz-1};
                // pSPARC->MpExp_exx = (MPEXP_OBJ *) malloc(sizeof(MPEXP_OBJ));
                // init_multipole_expansion(pSPARC, pSPARC->MpExp_exx, 
                //     pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, DMVertices, pSPARC->dmcomm);
                for (int i = 0; i < pSPARC->Nd; i++) { // attention: in the case of Dirichlet BC, 4*pi will be addad into RHS , not in the eigenvalues of Laplacian.
                    kron_lap->inv_eig[i] = sqrt(-kron_lap->inv_eig[i]);// minus and sqrt is addad at here
                }
            } else if (pSPARC->BC == 2) {
                // periodic BC needs singularity removal
                pSPARC->ExxDivFlag = 1;
                compute_pois_kron_cons_RPA(pSPARC, 1);
            }

            for (int omegaIndex = 0; omegaIndex < pRPA->Nomega; omegaIndex++) {
                if (pRPA->flagPreconditioner[omegaIndex]) {
                    pRPA->flagAppliedPrecond = 1;
                    break;
                }
            }
            if (pRPA->flagAppliedPrecond) {
                if (pSPARC->BC == 2) { // periodic BC
                    pRPA->pois_const_precond = (double *)malloc(sizeof(double) * pSPARC->Nd);
                    assert(pRPA->pois_const_precond != NULL);
                } else { // Dirichlet BC
                    pRPA->inv_eig_precond = (double *)malloc(sizeof(double) * pSPARC->Nd);
                    assert(pRPA->inv_eig_precond != NULL);
                }
            }
        } else {
            pRPA->deltaRhos_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            assert(pRPA->deltaRhos_kpt);
            pRPA->sprtNuDeltaVs_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm);
            assert(pRPA->sprtNuDeltaVs_kpt);
            pRPA->deltaPsis_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nd_d_dmcomm * pRPA->nNuChi0Eigscomm * 2);
            assert(pRPA->deltaPsis_kpt);
            if (pRPA->flagCOCGinitial) {
                pRPA->allXorb_kpt = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nspin_spincomm * pSPARC->Nkpts_kptcomm * pSPARC->Nd_d_dmcomm * pSPARC->Nstates);
                pRPA->allXorb_kPq = (double _Complex*)calloc(sizeof(double _Complex), pSPARC->Nspin_spincomm * pSPARC->Nkpts_kptcomm * pSPARC->Nd_d_dmcomm * pSPARC->Nstates);
                assert(pRPA->allXorb_kpt);
                pRPA->allLambdas = (double*)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nkpts_kptcomm * pSPARC->Nstates);
                pRPA->allLambdas_kPq = (double*)calloc(sizeof(double), pSPARC->Nspin_spincomm * pSPARC->Nkpts_kptcomm * pSPARC->Nstates);
            }

            for (int qptIndex = 0; qptIndex < pRPA->Nqpts_sym; qptIndex++) {
                pSPARC->kron_lap_exx[qptIndex] = (KRON_LAP *) malloc(sizeof(KRON_LAP)); // for K-points, it is necessary to modify this line.
                KRON_LAP* kron_lap = pSPARC->kron_lap_exx[qptIndex];
                init_kron_Lap(pSPARC, pSPARC->Nx, pSPARC->Ny, pSPARC->Nz, 
                    pSPARC->BCx, pSPARC->BCy, pSPARC->BCz, pRPA->q1[qptIndex], pRPA->q2[qptIndex], pRPA->q3[qptIndex], pSPARC->isGammaPoint, kron_lap);
            }
            pSPARC->ExxDivFlag = 1;
            compute_pois_kron_cons_RPA(pSPARC, pRPA->Nqpts_sym);
            
        }
    }
    if (pRPA->flagSQ||pRPA->flagLQ) {
        setup_HpDiagComm_RPASQ(pRPA, pSPARC->eig_paral_blksz, pSPARC->isGammaPoint, 1);
        if (pRPA->HpDiagCommIndex != -1) {
            pRPA->diagFHps = (double*)calloc(sizeof(double), pRPA->nHpDiagComm);
        }
    }
    if ((pRPA->flagSQ||pRPA->flagLQ) && (pRPA->printEigs)) {
        printf("Cannot print eigenvalues of nu chi0 when SQ is selected. The flag is invalid.\n");
        pRPA->printEigs = 0;
    }
    if (pRPA->printEigs) {
        snprintf(pRPA->filename_outEig, L_STRING + 3, "%sEig", pRPA->filename_out);
    }

    if (!rank) {
        write_output_init(pSPARC);
        write_settings(pRPA, pSPARC->isGammaPoint, pSPARC->Nspin_spincomm, pSPARC->Nstates, pSPARC->Nd_d_dmcomm);
    }
}

void RPA_Input_MPI_create(MPI_Datatype *RPA_INPUT_MPI) {
    RPA_INPUT_OBJ rpa_input_tmp;
    MPI_Datatype RPA_TYPES[N_MEMBR_RPA] =   {MPI_INT, MPI_INT,
                                    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
                                    MPI_DOUBLE,
                                    MPI_DOUBLE, MPI_DOUBLE,
                                    MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR, MPI_CHAR};
    int blens[N_MEMBR_RPA] = {15, 15,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1,1, 1, 1, 1,
                     15,
                     1, 1,
                     L_STRING, L_STRING, L_STRING, L_STRING, L_STRING};
    // calculating offsets in an architecture independent manner
    MPI_Aint addr[N_MEMBR_RPA],disps[N_MEMBR_RPA], base;
    int i = 0;
    MPI_Get_address(&rpa_input_tmp, &base);
    // int array type
    MPI_Get_address(&rpa_input_tmp.SternBlockSize, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagPreconditioner, addr + i++);
    // int type
    MPI_Get_address(&rpa_input_tmp.npnuChi0Neig, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npspin, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npkpt, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npband, addr + i++);
    MPI_Get_address(&rpa_input_tmp.npHpDiagSQ, addr + i++);
    MPI_Get_address(&rpa_input_tmp.nuChi0Neig, addr + i++);
    MPI_Get_address(&rpa_input_tmp.nplLanczosSQ, addr + i++);
    MPI_Get_address(&rpa_input_tmp.Nomega, addr + i++);
    MPI_Get_address(&rpa_input_tmp.maxitFiltering, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagPQ, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagCOCGinitial, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagLQ, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagNUM_LQ_ITER, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagWRITE_DIAGS, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagWRITE_NUCHI0_MULT_VECTORS_INFO, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagINTERP_MODE, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagNUM_COARSE_SKIP, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagSQ, addr + i++);
    MPI_Get_address(&rpa_input_tmp.flagCyclicPermute, addr + i++);
    MPI_Get_address(&rpa_input_tmp.printEigs, addr + i++);
    // double array type
    MPI_Get_address(&rpa_input_tmp.Icoeff, addr + i++);
    // double type
    MPI_Get_address(&rpa_input_tmp.tol_ErpaConverge, addr + i++);
    MPI_Get_address(&rpa_input_tmp.sternRelativeResTol, addr + i++);
    // char[] type
    MPI_Get_address(&rpa_input_tmp.filename, addr + i++);
    MPI_Get_address(&rpa_input_tmp.filename_out, addr + i++);
    MPI_Get_address(&rpa_input_tmp.InDensTCubFilename, addr + i++);
    MPI_Get_address(&rpa_input_tmp.InDensUCubFilename, addr + i++);
    MPI_Get_address(&rpa_input_tmp.InDensDCubFilename, addr + i++);
    for (i = 0; i < N_MEMBR_RPA; i++) {
        disps[i] = addr[i] - base;
    }

    MPI_Type_create_struct(N_MEMBR_RPA, blens, disps, RPA_TYPES, RPA_INPUT_MPI);
    MPI_Type_commit(RPA_INPUT_MPI);
}

void set_RPA_defaults(RPA_INPUT_OBJ *pRPA_Input, int Nstates, int Nd) {
    pRPA_Input->npnuChi0Neig = -1; 
    pRPA_Input->npspin = 0;
    pRPA_Input->npkpt = 0;
    pRPA_Input->npband = 0;
    pRPA_Input->npHpDiagSQ = 0;
    strncpy(pRPA_Input->filename, "UNDEFINED",sizeof(pRPA_Input->filename));  
    strncpy(pRPA_Input->filename_out, "UNDEFINED",sizeof(pRPA_Input->filename_out)); 
    strncpy(pRPA_Input->InDensTCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensTCubFilename));
    strncpy(pRPA_Input->InDensUCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensUCubFilename));
    strncpy(pRPA_Input->InDensDCubFilename, "UNDEFINED",sizeof(pRPA_Input->InDensDCubFilename));
    pRPA_Input->nuChi0Neig = (50*Nstates > Nd) ? 50*Nstates : Nd;
    pRPA_Input->nplLanczosSQ = 6;
    pRPA_Input->Nomega = 8;
    for (int i = 0; i < 15; i++) {
        pRPA_Input->SternBlockSize[i] = 1; 
        pRPA_Input->Icoeff[i] = 1e-4;
        pRPA_Input->flagPreconditioner[i] = 0;
    }
    pRPA_Input->tol_ErpaConverge = 5e-4;
    pRPA_Input->maxitFiltering = 20;
    pRPA_Input->sternRelativeResTol = 1e-2;
    pRPA_Input->flagPQ = 0;
    pRPA_Input->flagCOCGinitial = 1;
    pRPA_Input->flagSQ = 0;
    pRPA_Input->flagLQ = 0;
    pRPA_Input->flagNUM_LQ_ITER  = 0;
    pRPA_Input->flagWRITE_DIAGS  = 0;
    pRPA_Input->flagWRITE_NUCHI0_MULT_VECTORS_INFO = 0;
    pRPA_Input->flagINTERP_MODE = 0;
    pRPA_Input->flagNUM_COARSE_SKIP = 0;
    pRPA_Input->flagCyclicPermute = 0;
    pRPA_Input->printEigs = 0;
}

void read_RPA_inputs(RPA_INPUT_OBJ *pRPA_Input) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char input_filename[L_STRING + 4], str[L_STRING];
    snprintf(input_filename, L_STRING + 4, "%s.rpa", pRPA_Input->filename);
    
    FILE *input_fp = fopen(input_filename,"r");
    
    if (input_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",input_filename);
        print_usage();
        exit(EXIT_FAILURE);
    }

    while (!feof(input_fp)) {
        int count = fscanf(input_fp,"%s",str);
        if (count < 0) continue;  // for some specific cases
        
        // enable commenting with '#'
        if (str[0] == '#' || str[0] == '\n'|| strcmpi(str,"undefined") == 0) {
            fscanf(input_fp, "%*[^\n]\n"); // skip current line
            continue;
        }
        if (strcmpi(str,"NP_NUCHI_EIGS_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npnuChi0Neig);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_SPIN_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npspin);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_KPOINT_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npkpt);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_BAND_PARAL_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npband);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"NP_DIAG_SQ:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->npHpDiagSQ);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"N_NUCHI_EIGS:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->nuChi0Neig);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"N_OMEGA:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->Nomega);
            if ((pRPA_Input->Nomega < 0) || (pRPA_Input->Nomega > 15)) {
                printf("Please put a valid N_OMEGA in .rpa input file! N_OMEGA should be integer between [1, 15].\n");
                exit(EXIT_FAILURE);
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"STERN_BLOCK_SIZE:") == 0) {
            if (pRPA_Input->Nomega < 0) {
                printf("Please put the input STERN_BLOCK_SIZE below the input N_OMEGA. N_OMEGA must be set manually in .rpa input file!\n");
                exit(EXIT_FAILURE);
            }
            for (int i = 0; i < pRPA_Input->Nomega; i++) {
                int result = fscanf(input_fp,"%d",&pRPA_Input->SternBlockSize[i]);
                if (!result) {
                    printf("The input amount of STERN_BLOCK_SIZE is fewer than N_OMEGA. Please check the input again!\n");
                    exit(EXIT_FAILURE);
                }
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_ERPA:") == 0) {
            fscanf(input_fp,"%lf",&pRPA_Input->tol_ErpaConverge);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"TOL_STERN_RES:") == 0) {
            fscanf(input_fp,"%lf",&pRPA_Input->sternRelativeResTol);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"MAXIT_FILTERING:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->maxitFiltering);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FLAG_PQ_OPERATOR:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->flagPQ);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FLAG_COCGINITIAL:") == 0) {
            fscanf(input_fp,"%d",&pRPA_Input->flagCOCGinitial);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FLAG_PRECONDITION:") == 0) {
            if (pRPA_Input->Nomega < 0) {
                printf("Please put the input FLAG_PRECONDITION: below the input N_OMEGA. N_OMEGA must be set manually in .rpa input file!\n");
                exit(EXIT_FAILURE);
            }
            for (int i = 0; i < pRPA_Input->Nomega; i++) {
                int result = fscanf(input_fp,"%d",&pRPA_Input->flagPreconditioner[i]);
                if (!result) {
                    printf("The input amount of FLAG_PRECONDITION: is fewer than N_OMEGA. Please check the input again!\n");
                    exit(EXIT_FAILURE);
                }
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"PRECOND_ICOEFF:") == 0) {
            if (pRPA_Input->Nomega < 0) {
                printf("Please put the input PRECOND_ICOEFF below the input N_OMEGA. N_OMEGA must be set manually in .rpa input file!\n");
                exit(EXIT_FAILURE);
            }
            for (int i = 0; i < pRPA_Input->Nomega; i++) {
                int result = fscanf(input_fp,"%lf",&pRPA_Input->Icoeff[i]);
                if (!result) {
                    printf("The input amount of PRECOND_ICOEFF is fewer than N_OMEGA. Please check the input again!\n");
                    exit(EXIT_FAILURE);
                }
            }
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"LQ_FLAG_RPA:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->flagLQ);
            fscanf(input_fp, "%*[^\n]\n");
            pRPA_Input->npspin=1;
            pRPA_Input->npkpt=1;
            pRPA_Input->npband=1;
            pRPA_Input->npHpDiagSQ=1;//size;
            pRPA_Input->nplLanczosSQ=1;  
        } else if (strcmpi(str,"NUM_LQ_ITER:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->flagNUM_LQ_ITER);
            fscanf(input_fp, "%*[^\n]\n"); 
        } else if (strcmpi(str,"WRITE_DIAGS:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->flagWRITE_DIAGS);
            fscanf(input_fp, "%*[^\n]\n");         
        } else if (strcmpi(str,"WRITE_NUCHI0_MULT_VECTORS_INFO:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->flagWRITE_NUCHI0_MULT_VECTORS_INFO);
            fscanf(input_fp, "%*[^\n]\n");          
        } else if (strcmpi(str,"INTERP_MODE:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->flagINTERP_MODE);
            fscanf(input_fp, "%*[^\n]\n");          
        } else if (strcmpi(str,"NUM_COARSE_SKIP:") == 0) {
            fscanf(input_fp,"%d", &pRPA_Input->flagNUM_COARSE_SKIP);
            fscanf(input_fp, "%*[^\n]\n");   
        } else if (strcmpi(str,"SQ_FLAG_RPA:") == 0) {    
            fscanf(input_fp,"%d",&pRPA_Input->flagSQ);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"SQ_NPL_RPA:") == 0) {    
            fscanf(input_fp,"%d",&pRPA_Input->nplLanczosSQ);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"FLAG_CYCLIC_PERMUTE:") == 0) {    
            fscanf(input_fp,"%d",&pRPA_Input->flagCyclicPermute);
            fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmpi(str,"INPUT_DENS_FILE:") == 0) {
            char inputDensFnames[3][L_STRING]; // at most 3 file names
            int nInputDensFname = readStringInputsFromFile(input_fp, 3, inputDensFnames);
            if (nInputDensFname == 1) {
                strncpy(pRPA_Input->InDensTCubFilename, inputDensFnames[0], L_STRING);
            } else if (nInputDensFname == 3) {
                strncpy(pRPA_Input->InDensTCubFilename, inputDensFnames[0], L_STRING);
                strncpy(pRPA_Input->InDensUCubFilename, inputDensFnames[1], L_STRING);
                strncpy(pRPA_Input->InDensDCubFilename, inputDensFnames[2], L_STRING);
            } else {
                printf(RED "[FATAL] Density file names not provided properly! (Provide 1 file w/o spin or 3 files with spin)\n" RESET);
                exit(EXIT_FAILURE);
            }

            #ifdef DEBUG
            if (nInputDensFname >= 0) {
                printf("Density file names read = ([");
                for (int i = 0; i < nInputDensFname; i++) {
                    if (i == 0) printf("%s", inputDensFnames[i]);
                    else printf(", %s", inputDensFnames[i]);
                }
                printf("], %d)\n", nInputDensFname);
            }
            printf("Total Dens file name: %s\n", pRPA_Input->InDensTCubFilename);
            printf("Dens_up file name: %s\n", pRPA_Input->InDensUCubFilename);
            printf("Dens_dw file name: %s\n", pRPA_Input->InDensDCubFilename);
            #endif
        } else if (strcmpi(str,"FLAG_PRINT_EIGS:") == 0) {    
            fscanf(input_fp,"%d",&pRPA_Input->printEigs);
            fscanf(input_fp, "%*[^\n]\n");
        } else {
            printf("\nCannot recognize input variable identifier: \"%s\"\n",str);
            exit(EXIT_FAILURE);
        }
    }

    fclose(input_fp);
}

void RPA_copy_inputs(RPA_OBJ *pRPA, RPA_INPUT_OBJ *pRPA_Input) {
    pRPA->npnuChi0Neig = pRPA_Input->npnuChi0Neig; // the validity of np settings will be checked in function Setup_Comms_RPA
    pRPA->npspin = pRPA_Input->npspin;
    pRPA->npkpt = pRPA_Input->npkpt;
    pRPA->npband = pRPA_Input->npband;
    pRPA->npHpDiagSQ = pRPA_Input->npHpDiagSQ;
    strncpy(pRPA->filename, pRPA_Input->filename,sizeof(pRPA_Input->filename)); 
    strncpy(pRPA->filename_out, pRPA_Input->filename_out,sizeof(pRPA_Input->filename_out)); 
    strncpy(pRPA->InDensTCubFilename, pRPA_Input->InDensTCubFilename,sizeof(pRPA_Input->InDensTCubFilename));
    strncpy(pRPA->InDensUCubFilename, pRPA_Input->InDensUCubFilename,sizeof(pRPA_Input->InDensUCubFilename));
    strncpy(pRPA->InDensDCubFilename, pRPA_Input->InDensDCubFilename,sizeof(pRPA_Input->InDensDCubFilename));
    pRPA->nuChi0Neig = pRPA_Input->nuChi0Neig;
    pRPA->nplLanczosSQ = pRPA_Input->nplLanczosSQ;
    pRPA->Nomega = pRPA_Input->Nomega;
    for (int i = 0; i < 15; i++) {
        pRPA->SternBlockSize[i] = pRPA_Input->SternBlockSize[i];
        pRPA->flagPreconditioner[i] = pRPA_Input->flagPreconditioner[i];
        pRPA->Icoeff[i] = pRPA_Input->Icoeff[i];
    }
    pRPA->maxitFiltering = pRPA_Input->maxitFiltering;
    pRPA->tol_ErpaConverge = pRPA_Input->tol_ErpaConverge;
    pRPA->sternRelativeResTol = pRPA_Input->sternRelativeResTol;
    pRPA->flagPQ = pRPA_Input->flagPQ;
    pRPA->flagCOCGinitial = pRPA_Input->flagCOCGinitial;
    pRPA->flagLQ = pRPA_Input->flagLQ;
    pRPA->flagNUM_LQ_ITER = pRPA_Input->flagNUM_LQ_ITER;
    pRPA->flagWRITE_DIAGS = pRPA_Input->flagWRITE_DIAGS;
    pRPA->flagWRITE_NUCHI0_MULT_VECTORS_INFO = pRPA_Input->flagWRITE_NUCHI0_MULT_VECTORS_INFO;
    pRPA->flagINTERP_MODE = pRPA_Input->flagINTERP_MODE;
    pRPA->flagNUM_COARSE_SKIP = pRPA_Input->flagNUM_COARSE_SKIP;
    pRPA->flagSQ = pRPA_Input->flagSQ; 
    pRPA->flagCyclicPermute = pRPA_Input->flagCyclicPermute;
    if (!((pRPA->nuChi0Neig > 0) && (pRPA->Nomega > 0) && (pRPA->maxitFiltering > 0))) {
        printf("\nN_NUCHI_EIGS, N_OMEGA, MAXIT_FILTERING need to be positive integer.\n");
        exit(EXIT_FAILURE);
    }
    pRPA->printEigs = pRPA_Input->printEigs;
    // if (pRPA->flagPQ && pRPA->flagCOCGinitial) {
    //     printf("\nCurrently the code does not support using PQ and COCG initial guess together.\n");
    //     exit(EXIT_FAILURE);
    // }
}

void set_omegas(double *omega, double *omega01, double *omegaWts, int Nomega) {
    // from ABINIT, Gauss Legendre coefficient
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double tolerance = 1e-13;
    double length = (1.0 - 0.0) / 2;
    double mean = (1.0 + 0.0) / 2;
    double z;
    int flag;
    double p1, p2, p3, pp, z1;
    for (int index = 1; index < Nomega + 1; index++) {
        z = cos(M_PI*((double)index - 0.25) / ((double)Nomega + 0.5));
        flag = 1;
        while (flag) {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 1; j < Nomega + 1; j++) {
                p3 = p2;
                p2 = p1;
                p1 = ((2.0*(double)j - 1.0)*z*p2 - ((double)j - 1.0)*p3) / (double)j;
            }
            pp = (double)Nomega*(p2 - z*p1) / (1.0 - z*z);
            z1 = z;
            z = z1 - p1/pp;
            flag = (fabs(z - z1) > tolerance);
        }
        omega01[index - 1] = mean - length*z;
        omega01[Nomega - index] = mean + length*z;
        omegaWts[index - 1] = 2.0*length / ((1.0 - z*z) * (pp*pp));
        omegaWts[Nomega - index] = omegaWts[index - 1];
    }
    for (int index = 0; index < Nomega; index++) {
        omega[index] = 1.0 / omega01[index] - 1.0;
    }
    #ifdef DEBUG
    if (!rank) {
        printf("integration point omega and weight are\n");
        for (int index = 0; index < Nomega; index++)
            printf("omega %9.6f, omega01 %9.6f, weight %9.6f\n", omega[index], omega01[index], omegaWts[index]);
    }
    #endif
}

void write_settings(RPA_OBJ *pRPA, int isGammaPoint, int nspin, int nstates, int Nd_d_dmcomm) {
    FILE *output_fp = fopen(pRPA->filename_out,"a");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",pRPA->filename_out);
        exit(EXIT_FAILURE);
    }
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                   RPA Settings, Version Apr 17, 2025                      \n");
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"N_NUCHI_EIGS: %d\n",pRPA->nuChi0Neig);
    fprintf(output_fp,"N_OMEGA: %d\n",pRPA->Nomega);
    fprintf(output_fp,"STERN_BLOCK_SIZE: ");
    for (int i = 0; i < pRPA->Nomega; i++) {
        fprintf(output_fp,"%d ",pRPA->SternBlockSize[i]);
    }
    fprintf(output_fp,"\n");
    if (pRPA->flagSQ) {
        fprintf(output_fp,"TOL_ERPA: %.3E\n",pRPA->tol_ErpaConverge);
    }
    fprintf(output_fp,"TOL_STERN_RES: %.3E\n",pRPA->sternRelativeResTol);
    fprintf(output_fp,"FLAG_PQ_OPERATOR: %d\n",pRPA->flagPQ);
    fprintf(output_fp,"FLAG_COCGINITIAL: %d\n",pRPA->flagCOCGinitial);
    fprintf(output_fp,"FLAG_PRECONDITION: ");
    for (int i = 0; i < pRPA->Nomega; i++) {
        fprintf(output_fp,"%d ", pRPA->flagPreconditioner[i]);
    }
    fprintf(output_fp,"\n");
    fprintf(output_fp,"MAXIT_FILTERING: %d\n",pRPA->maxitFiltering);
    fprintf(output_fp,"INPUT_DENS_FILE: %s\n",pRPA->InDensTCubFilename);
    fprintf(output_fp,"FLAG_CYCLIC_PERMUTE: %d\n",pRPA->flagCyclicPermute);
    if (pRPA->flagSQ) {
        fprintf(output_fp,"SQ_FLAG_RPA: %d\n",pRPA->flagSQ);
        fprintf(output_fp,"SQ_NPL_RPA: %d\n",pRPA->nplLanczosSQ);
    }
    if (pRPA->flagAppliedPrecond) {
        fprintf(output_fp,"PRECOND_ICOEFF: ");
        for (int i = 0; i < pRPA->Nomega; i++) {
            fprintf(output_fp,"%.2E ",pRPA->Icoeff[i]);
        }
        fprintf(output_fp,"\n");
    }
    fprintf(output_fp,"FLAG_PRINT_EIGS: %d\n", pRPA->printEigs);


    fprintf(output_fp,"LQ_FLAG_RPA: %d\n",pRPA->flagLQ);
    if(pRPA->flagLQ>=1){
        fprintf(output_fp,"NUM_LQ_ITER: %d\n",pRPA->flagNUM_LQ_ITER);
        fprintf(output_fp,"WRITE_DIAGS: %d\n",pRPA->flagWRITE_DIAGS);
        fprintf(output_fp,"WRITE_NUCHI0_MULT_VECTORS_INFO: %d\n",pRPA->flagWRITE_NUCHI0_MULT_VECTORS_INFO);
        fprintf(output_fp,"INTERP_MODE: %d\n",pRPA->flagINTERP_MODE);
        if(pRPA->flagINTERP_MODE>=1){
        fprintf(output_fp,"NUM_COARSE_SKIP: %d\n",pRPA->flagNUM_COARSE_SKIP);
        }
    }


    
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"                         RPA Parallelization                               \n");
    fprintf(output_fp,"***************************************************************************\n");
    fprintf(output_fp,"NP_NUCHI_EIGS_PARAL_RPA: %d\n",pRPA->npnuChi0Neig);
    fprintf(output_fp,"NP_SPIN_PARAL_RPA: %d\n",pRPA->npspin);
    fprintf(output_fp,"NP_KPOINT_PARAL_RPA: %d\n",pRPA->npkpt);
    fprintf(output_fp,"NP_BAND_PARAL_RPA: %d\n",pRPA->npband);
    if (pRPA->flagSQ) {
        fprintf(output_fp,"NP_DIAG_SQ: %d\n",pRPA->npHpDiagSQ);
    }
    fprintf(output_fp,"***************************************************************************\n");
    int maxBlockSize = pRPA->SternBlockSize[0];
    for (int i = 1; i < pRPA->Nomega; i++) {
        if (pRPA->SternBlockSize[i] > maxBlockSize) {
            maxBlockSize = pRPA->SternBlockSize[i];
        }
    }
    long long memoryMB = 0;
    if (isGammaPoint) {
        memoryMB = 17154400 / 1000000 + (9370144+4340256) / 1000000 + (Nd_d_dmcomm*pRPA->nNuChi0Eigscomm*7*8) / 1000000
            + (Nd_d_dmcomm*maxBlockSize*(6*2 + 2)*8) / 1000000 + (Nd_d_dmcomm*nspin*nstates*8*pRPA->flagCOCGinitial) / 1000000
            + (pRPA->nr_Hp_BLCYC*pRPA->nc_Hp_BLCYC*8*4) / 1000000;
        if (pRPA->flagSQ) { 
            memoryMB += (pRPA->nrHpSQ*pRPA->ncHpSQ*8) / 1000000 + (pRPA->nplLanczosSQ*pRPA->nHpDiagComm*2*8) / 1000000
             + (pRPA->nrvk_BLCYC*pRPA->nHpDiagComm*3*8) / 1000000;
        }
        else if (pRPA->flagLQ){
            memoryMB+=(Nd_d_dmcomm*(pRPA->flagNUM_LQ_ITER+1)*8)/1000000.0+((pRPA->flagNUM_LQ_ITER*pRPA->flagNUM_LQ_ITER+2*pRPA->flagNUM_LQ_ITER)*8)/1000000.0;
        }
    } else {
        memoryMB = 17154400 / 1000000 + (9370144+4340256) / 1000000 + (Nd_d_dmcomm*pRPA->nNuChi0Eigscomm*7*16) / 1000000
            + (Nd_d_dmcomm*maxBlockSize*(6*2 + 2)*16) / 1000000 + (Nd_d_dmcomm*nspin*nstates*16*pRPA->flagCOCGinitial) / 1000000
            + (pRPA->nr_Hp_BLCYC*pRPA->nc_Hp_BLCYC*16*4) / 1000000;
        if (pRPA->flagSQ||pRPA->flagLQ) {//HAQUE TODO: Fix for Lanczos, which doesnt support k-point yet
            memoryMB += (pRPA->nrHpSQ*pRPA->ncHpSQ*16) / 1000000 + (pRPA->nplLanczosSQ*pRPA->nHpDiagComm*2*16) / 1000000
             + (pRPA->nrvk_BLCYC*pRPA->nHpDiagComm*3*16) / 1000000;
        }
    }
    if (memoryMB < 1000) {
        fprintf(output_fp,"Estimated memory usage in RPA calculation is %d MB\n", memoryMB);
    } else {
        fprintf(output_fp,"Estimated memory usage in RPA calculation is %d GB\n", memoryMB / 1000);
    }
    fclose(output_fp);
}

void compute_pois_kron_cons_RPA(SPARC_OBJ *pSPARC, int Nqpts_sym) {

    pSPARC->pois_const = (double *)malloc(sizeof(double) * Nqpts_sym*pSPARC->Nd);
    assert(pSPARC->pois_const != NULL);

    if (pSPARC->BC == 2) {
        // auxiliary function
        for (int qptIndex = 0; qptIndex < Nqpts_sym; qptIndex++) {
            KRON_LAP* kron_lap = pSPARC->kron_lap_exx[qptIndex];
            for (int i = 0; i < pSPARC->Nd; i++) {
                int constIndex = qptIndex*pSPARC->Nd + i;
                double G2 = -kron_lap->eig[i];     
                if (fabs(G2) > 1e-8) {
                    pSPARC->pois_const[constIndex] = sqrt(4*M_PI/G2); // only this value is used by RPA calculation, to compose \nu^{0.5}
                } else {
                    pSPARC->pois_const[constIndex] = sqrt(4*M_PI*pSPARC->const_aux); // only this value is used by RPA calculation, to compose \nu^{0.5}
                }
            }
        }

        // printf("%d th pois_const %.4E, its G2 %.4E\n", i, pSPARC->pois_const[i], G2);
    }
}
