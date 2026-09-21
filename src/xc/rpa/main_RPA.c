/**
 * @file    main.c
 * @brief   This file contains the main function for real-space RPA calculation
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

 /*
 Structure of RPA program
 main
 ├──initialization_RPA
 |  ├──Initialize_SPARC_before_SetComm
 |  ├──transfer_kpoints, recalculate_kpoints
 |  ├──RPA_Input_MPI_create, set_RPA_defaults, read_RPA_inputs, RPA_copy_inputs
 |  ├──set_kPq_lists, set_omegas
 |  ├──Setup_Comms_RPA
 |  ├──Initialize_SPARC_SetComm_after
 |  ├──setup_blacsComm_RPA
 |  ├──generate_permute_matrix
 |  └──write_settings
 ├──restore_electronicGroundState
 |  ├──restore_orbitals
 |  |  ├──read_orbitals_distributed_gamma_RPA
 |  |  └──read_orbitals_distributed_kpt_RPA
 |  ├──restore_electronDensity
 |  └──restore_eigval_occ
 ├──initialize_deltaVs
 ├──test_Hx_nuChi0
 ├──subspace_iteration_RPA──────────────────────────────────────────subspace_iteration_RPASQ
 |  ├──(if gamma point) collect_allXorb_allLambdas_gamma            ├──(if gamma point) collect_allXorb_allLambdas_gamma
 |  ├──project_tildeChi_general_eigProblem                          ├──project_tildeChi_update_deltaV
 |  |  ├──nuChi0_mult_vectors_gamma                                 |  ├──nuChi0_mult_vectors_gamma
 |  |  |  ├──sternheimer_eq_gamma                                   |  |  (refer to left...)
 |  |  |  |  ├──compute_pois_kron_cons_LapPrecond                   |  |
 |  |  |  |  └──sternheimer_solver_gamma                            |  |
 |  |  |  |     ├──Sternheimer_lhs                                  |  |
 |  |  |  |     ├──laplace_preconditioner                           |  |
 |  |  |  |     ├──set_initial_guess_deltaPsis                      |  |
 |  |  |  |     └──block_COCG                                       |  |
 |  |  |  ├──collect_deltaRho_gamma                                 |  |
 |  |  |  └──Calculate_sqrtNu_vecs_gamma                            |  |
 |  |  |                                                            |  ├──YT_multiply_Y_gamma
 |  |  ├──YT_multiply_Y_gamma                                       |  ├──Y_orth_permute_gamma
 |  |  ├──project_YT_nuChi0_Y_gamma                                 |  └──project_YT_nuChi0_Y_gamma
 |  |  ├──generalized_eigenproblem_solver_gamma                     ├──transfer_Hp_HpDiagComms
 |  |  └──subspace_rotation_for_newYs                               ├──SQ_trace_estimator
 |  |                                                               |  ├──Lanczos_decompose_Hp_group
 |  |                                                               |  └──diagonalize_T_get_diag_fHp
 |  |                                                               |
 |  ├──(if k-point) collect_allXorb_allLambdas_kpt                  ├──(if k-point) collect_allXorb_allLambdas_kpt
 |  ├──send_recv_allXorb_allLambdas_kPq                             ├──send_recv_allXorb_allLambdas_kPq
 |  ├──project_tildeChi_general_eigProblem_kpt                      ├──project_tildeChi_update_deltaV_kpt
 |  |  ├──nuChi0_mult_vectors_kpt                                   |  ├──nuChi0_mult_vectors_kpt
 |  |  |  ├──sternheimer_eq_kpt                                     |  |  (refer to left...)
 |  |  |  |  ├──compute_pois_kron_cons_LapPrecond                   |  |
 |  |  |  |  └──sternheimer_solver_kpt                              |  |
 |  |  |  |     ├──Sternheimer_lhs_kpt                              |  |
 |  |  |  |     ├──set_initial_guess_deltaPsis_kpt                  |  |
 |  |  |  |     └──block_CRM                                        |  |
 |  |  |  ├──collect_deltaRho_kpt                                   |  |
 |  |  |  └──Calculate_sqrtNu_vecs_kpt                              |  |
 |  |  |                                                            |  ├──project_YT_nuChi0_Y_kpt
 |  |  ├──YT_multiply_Y_kpt                                         |  ├──YT_multiply_Y_kpt
 |  |  ├──project_YT_nuChi0_Y_kpt                                   |  └──project_YT_nuChi0_Y_kpt
 |  |  ├──generalized_eigenproblem_solver_kpt                       ├──transfer_Hp_HpDiagComms_kpt
 |  |  └──subspace_rotation_for_newYs_kpt                           └──SQ_trace_estimator_kpt
 |  |                                                                  ├──Lanczos_decompose_Hp_group_kpt
 |  |                                                                  └──diagonalize_T_get_diag_fHp
 |  └──compute_ErpaTerm
 |
 ├──print_result
 |  └──print_eigs_nuchi0
 └──finalizeRPA
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "tools.h"

#include "main_RPA.h"
#include "initialization_RPA.h"
#include "restoreElectronicGroundState.h"
#include "prepare_PQ_operators.h"
#include "test_Hx_nuChi0_eigSolver.h"
#include "subspaceIter.h"
#include "subspaceIterSQ.h"
#include "lanczosQuadrature.h"
#include "printResult.h"
#include "finalization_RPA.h"

void main_RPA(SPARC_OBJ *pSPARC, int argc, char* argv[]) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    RPA_OBJ RPA; // save new variables needed for RPA calculation

    double t1, t2;
    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    initialize_RPA(pSPARC, &RPA, argc, argv);

    restore_electronicGroundState(pSPARC, RPA.nuChi0Eigscomm, RPA.nuChi0EigsBridgeComm, RPA.nuChi0EigscommIndex, RPA.rank0nuChi0EigscommInWorld, RPA.k1, RPA.k2, RPA.k3, RPA.kPqSymList, RPA.Nkpts_sym);

    // prepare_PQ_operators(&SPARC, &RPA);

    initialize_deltaVs(pSPARC, &RPA);

    int testFlag = 0;
    if (testFlag) {
        test_Hx_nuChi0(pSPARC, &RPA);
    }

    int testEigFlag = 0;
    if (testEigFlag) {
        test_eigSolver(pSPARC, &RPA);
    }

    for (int qptIndex = 0; qptIndex < RPA.Nqpts_sym; qptIndex++) {
        for (int omegaIndex = 0; omegaIndex < RPA.Nomega; omegaIndex++) {
            if(RPA.flagLQ){ 
                lanczos_quadrature_RPA(pSPARC, &RPA, qptIndex, omegaIndex);
            }
            else if (RPA.flagSQ) {
                subspace_iteration_RPASQ(pSPARC, &RPA, qptIndex, omegaIndex);
            } else {
                subspace_iteration_RPA(pSPARC, &RPA, qptIndex, omegaIndex);
            }
        }
    }

    print_result(&RPA, pSPARC->n_atom);

    finalize_RPA(pSPARC, &RPA);

    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("The RPA calculation took %.3f s.\n", t2 - t1); 
    }
}

