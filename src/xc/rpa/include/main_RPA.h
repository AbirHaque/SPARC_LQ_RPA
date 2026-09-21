/**
 * @file    main.h
 * @brief   This file contains the structure definitions for SPARC and RPA
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
#ifndef MAINRPA
#define MAINRPA

#include <mpi.h>
#include <complex.h>

#include "isddft.h"

typedef struct _RPA_OBJ {
    // MPI communicators and their parallelizing parameters
    MPI_Comm nuChi0Eigscomm; // communicator for dividing trial vectors, whose amount equals to number of desired eigs of nuChi0
    int npnuChi0Neig;
    int nuChi0EigscommIndex;
    int nNuChi0Eigscomm;
    int nuChi0EigsStartIndex;
    int nuChi0EigsEndIndex;
    int rank0nuChi0EigscommInWorld;
    MPI_Comm nuChi0EigsBridgeComm; // communicator for linking ALL processors having the same rank of nuChi0EigsComm. Every nuChi0EigsComm needs
    // a complete set of eigenvalues, eigenvectors and occupations from K-S DFT calculation
    int nuChi0EigsBridgeCommIndex; // which equals to rank of the processor in nuChi0Eigscomm
    MPI_Comm nuChi0BlacsComm; // communicator for using blacs to do calculation between nuChi0Eigscomms
    int ictxt_blacs;    // fortran handle for the context corresponding to blacscomm (for ScaLAPACK) (LOCAL)
    int ictxt_blacs_topo;  // fortran handle for the Cartesian topology within the ictxt_blacs context (for ScaLAPACK) (LOCAL)
    int nprow_ictxt_blacs_topo; // number of rows in the ScaLAPACK context ictxt_blacs_topo (LOCAL)
    int npcol_ictxt_blacs_topo; // number of cols in the ScaLAPACK context ictxt_blacs_topo (LOCAL)
    int desc_orbitals[9];  // ScaLAPACK descriptor for storage of the orbitals on each blacscomm
    int desc_orb_BLCYC[9]; // descriptor for BLOCK CYCLIC distribution of the orbitals on each blacscomm
    int desc_permute[9];
    double *permute;
    double _Complex *permute_kpt;
    int rPermute;
    int cPermute;
    double *Ys_phi_BLCYC;
    double _Complex *Ys_kpt_phi_BLCYC;
    int desc_Hp_BLCYC[9];  // descriptor for BLOCK CYCLIC distribution of the projected Hamiltonian on each ictxt_blacs_topo
    int desc_Mp_BLCYC[9];  // descriptor for BLOCK CYCLIC distribution of the overlap matrix on each ictxt_blacs_topo
    int desc_Q_BLCYC[9];   // descriptor for BLOCK CYCLIC distribution of the eigenvectors on each ictxt_blacs_topo
    int desc_permute_BLCYC[9];
    double *permute_BLCYC;
    double _Complex *permute_BLCYC_kpt;
    int nr_orb_BLCYC;          // number of rows of the local distributed orbitals owned by the process (LOCAL)
    int nc_orb_BLCYC;          // number of cols of the local distributed orbitals owned by the process (LOCAL)
    int nr_Hp_BLCYC;           // number of rows of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int nc_Hp_BLCYC;           // number of cols of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int nr_Mp_BLCYC;           // number of rows of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int nc_Mp_BLCYC;           // number of cols of the local distributed projected Hamiltonian owned by the process (LOCAL)
    int nr_Q_BLCYC;            // number of rows of the local distributed subspace eigenvectors owned by the process (LOCAL)
    int nc_Q_BLCYC;            // number of cols of the local distributed subspace eigenvectors owned by the process (LOCAL)
    double *Hp;                   // projected Hamiltonian matrix: Hp = Psi' * H * Psi (LOCAL)
    double *Mp;                   // projected mass matrix: Mp = Psi' * Psi (LOCAL)
    double *Q;                    // eigenvectors of the generalized eigenproblem: Hp * Q_i  = lambda_i * Mp * Q_i
    double _Complex *Hp_kpt;                   // projected Hamiltonian matrix: Hp = Psi' * H * Psi (LOCAL)
    double _Complex *Mp_kpt;                   // projected mass matrix: Mp = Psi' * Psi (LOCAL)
    double _Complex *Q_kpt;                    // eigenvectors of the generalized eigenproblem: Hp * Q_i  = lambda_i * Mp * Q_i
    int eig_useLAPACK;                // flag for using LAPACK_dsygv to solve subspace eigenproblem
    int eig_paral_subdims[2];
    int eig_paral_maxnp;
    // SPARC parallelizing parameters to be used in RPA calculation
    int npspin;         // number of spin communicators
    int npkpt;          // number of processes for paral. over k-points
    int npband;         // number of processes for paral. over bands
    // SQ parallel param
    int npHpDiagSQ;
    // other settings for RPA computation
    char filename[L_STRING];
    char filename_out[L_STRING];
    char filename_outEig[L_STRING + 3];
    char InDensTCubFilename[L_STRING];
    char InDensUCubFilename[L_STRING];
    char InDensDCubFilename[L_STRING];
    char timeRecordname[L_STRING];
    int maxitFiltering;
    double tol_ErpaConverge;
    double sternRelativeResTol;
    int flagPQ;
    int flagCOCGinitial;
    int flagPreconditioner[15];
    int flagAppliedPrecond;
    int flagLQ;
    int flagNUM_LQ_ITER;
    int flagWRITE_DIAGS;
    int flagWRITE_NUCHI0_MULT_VECTORS_INFO;
    int flagINTERP_MODE;
    int flagNUM_COARSE_SKIP;
    int flagSQ;
    int flagCyclicPermute;
    // q-points, which is k-point grid without shift after symmetry reduction
    int Nqpts_sym; // amount of q-points
    double *qptWts;
    double *q1;
    double *q2;
    double *q3;
    // omegas, which is integral points and weights from [0, +\infty]
    int Nomega;
    double *omega;
    double *omega01;
    double *omegaWts;
    int SternBlockSize[15];
    // amount of eigenvalues of \nu\chi0 to be solved
    int nuChi0Neig;
    double *RRnuChi0Eigs;
    double *RRnuChi0EigVecs;
    double *ErpaTerms; // save Erpa(qpt, omega) terms, length Nqpts_sym * Nomega
    int nplLanczosSQ;
    // symmetric reduced k-point grid, saved for reading orbitals
    int Nkpts_sym;
    double *kptWts;
    double *k1;
    double *k2;
    double *k3;
    int **kPqSymList;
    int **kPqList;
    int **kMqList;
    // variables recording nearby bands for composing P and Q
    int *nearbyBandIndicesGamma;
    int neighborBandStartEnd[2];
    int *neighborBandIndicesGamma;
    double *neighborBandsGamma;
    double *allEpsilonsGamma;
    // save \Delta V s, \Delta\psi and \Delta\rho s
    double *Ys; // in dmcomm, save medium variable Y for Chebyshev filtering; length is pSPARC->Nd_d_dmcomm * nNuChi0Eigscomm
    double _Complex *Ys_kpt; // in dmcomm, save medium variable Y for Chebyshev filtering; length is pSPARC->Nd_d_dmcomm * nNuChi0Eigscomm
    double *deltaVs; // Its length is pSPARC->Nd_d_dmcomm*nNuChi0Eigscomm
    double _Complex *deltaVs_kpt; // Its length is pSPARC->Nd_d_dmcomm*nNuChi0Eigscomm
    double *sprtNuDeltaVs;
    double _Complex *sprtNuDeltaVs_kpt;
    double *deltaRhos; // in dmcomm, save sum of \psi_n^*(\Delta\psi_n) over bandcomm n; then AllReduce over bandcomm, kptcomm and spincomm. Its length is pSPARC->Nd_d_dmcomm * nNuChi0Eigscomm
    double _Complex *deltaRhos_kpt; // in dmcomm, save sum of \psi_n^*(\Delta\psi_n) over bandcomm n; then AllReduce over bandcomm, kptcomm and spincomm. Its length is pSPARC->Nd_d_dmcomm * nNuChi0Eigscomm
    double *deltaPsisReal; // in dmcomm, save real part of (\Delta\psi_n) of the current band, kpt and spin. Its length is pSPARC->Nd_d_dmcomm * nNuChi0Eigscomm
    double *deltaPsisImag; // in dmcomm, save imag part of (\Delta\psi_n) of the current band, kpt and spin. Its length is pSPARC->Nd_d_dmcomm * nNuChi0Eigscomm
    double _Complex *deltaPsis_kpt; // in dmcomm, save (\Delta\psi_n) of the current band, kpt, spin and \Delta V. Its length is pSPARC->Nd_d_dmcomm * 2
    // save bands for COCG initial guess
    int NoccupiedOrbital;
    double *allXorb;
    double _Complex *allXorb_kpt;
    double _Complex *allXorb_kPq;
    double *allLambdas;
    double *allLambdas_kPq;
    // Laplace preconditioner
    double *pois_const_precond; // for periodic BC
    double *inv_eig_precond; // for Dirichlet BC
    double Icoeff[15];

    // SQ related variables
    MPI_Comm HpDiagComm; // communicator for computing diagonal entries of f(Hp) by SQ
    int HpDiagCommIndex;
    int nHpDiagComm;
    int HpDiagStartIndex;
    int HpDiagEndIndex;
    int rank0HpDiagCommInWorld;
    int ctxtWorld;
    MPI_Comm HpDiagBridgeComm;
    int HpDiagBridgeCommIndex;
    int ictxt_HpSQ;
    int nrHpSQ;
    int ncHpSQ;
    int desc_HpSQ[9];
    double *HpSQ;
    double _Complex *HpSQ_kpt;
    int nrvk;
    int desc_vk[9];
    int vkStart;
    int vkEnd;
    int nrvk_BLCYC;
    int desc_vk_BLCYC[9];
    double *diagFHps;
    MPI_Comm commWorld; // MPI_COMM_WORLD has problem when doing bcast in for or while loop

    // timing counter for sternheimer equation for every qpt and omega
    double sternheimerTimeRecorder[5]; 
    #ifdef DEBUG
    // timing counter for operator multiplication
    double sumSqrtNuMulVecsTime;
    double sumSternheimerTime;
    double sumCollectdRhoTime;
    // timing counter for subspace iteration
    double sumOperatorMulTime; // total time for all nuChi0EigsComm finish XV
    double sumOperatorMulTime_nuChi0EigsComm; // time for this nuChi0EigsComm finish XV
    double sumYT_mul_YTime;
    double sumYT_operator_YTime;

    double sumEigTime;
    double sumRotationTime;

    double sum_orthonormalTime;
    double sumTransfer_HpTime;
    double sumSpectral_quadratureTime;
    #endif

    int printEigs;
} RPA_OBJ;

typedef struct _RPA_INPUT_OBJ {
    // MPI parallelizing parameters
    int npnuChi0Neig;
    // SPARC parallelizing parameters to be used in RPA calculation
    int npspin;         // number of spin communicators
    int npkpt;          // number of processes for paral. over k-points
    int npband;         // number of processes for paral. over bands
    // SQ parallel param
    int npHpDiagSQ;
    // other settings for RPA computation
    int nuChi0Neig;
    int Nomega;
    int nplLanczosSQ;
    int SternBlockSize[15];
    int maxitFiltering;
    int flagPQ;
    int flagCOCGinitial;
    int flagPreconditioner[15];
    int flagLQ;
    int flagNUM_LQ_ITER;
    int flagWRITE_DIAGS;
    int flagWRITE_NUCHI0_MULT_VECTORS_INFO;
    int flagINTERP_MODE;
    int flagNUM_COARSE_SKIP;
    int flagSQ;
    int flagCyclicPermute;
    double tol_ErpaConverge;
    double sternRelativeResTol;
    double Icoeff[15];
    char filename[L_STRING];
    char filename_out[L_STRING]; 
    char InDensTCubFilename[L_STRING];
    char InDensUCubFilename[L_STRING];
    char InDensDCubFilename[L_STRING];

    int printEigs;
} RPA_INPUT_OBJ;

void main_RPA(SPARC_OBJ *pSPARC, int argc, char* argv[]);


#endif