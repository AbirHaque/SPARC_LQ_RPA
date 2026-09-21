/**
 * @file    restoreElectronicGroundState.c
 * @brief   This file saves functions for restoring all necessary variables for generating Hamiltonian
 *          and all data for electron density and orbitals
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 

#include <complex.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "electronicGroundState.h"
#include "electrostatics.h"
#include "nlocVecRoutines.h"
#include "spinOrbitCoupling.h"
#include "orbitalElecDensInit.h"
#include "tools.h"
#include "exchangeCorrelation.h"
#include "parallelization.h"
#include "mGGAtauTransferTauVxc.h"

#include "restoreElectronicGroundState.h"

void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld,
     double *symk1, double *symk2, double *symk3, int **kPqSymList, int Nkpts_sym) {
    pSPARC->Xorb = NULL;
    pSPARC->Yorb = NULL;
    pSPARC->Xorb_kpt = NULL;
    pSPARC->Yorb_kpt = NULL;
    
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    
#ifdef DEBUG
    if (rank == 0) printf("Start ground-state calculation.\n");
#endif
    
    #ifdef DEBUG
    double t1, t2;
    if (rank == 0) printf("Calculating electron density ... \n");
    t1 = MPI_Wtime();
#endif
    
    // find atoms that influence the process domain
    GetInfluencingAtoms(pSPARC);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nFinding influencing atoms took %.3f ms\n",(t2-t1)*1000); 
    t1 = MPI_Wtime();
#endif
    
    // calculate pseudocharge density b
    Generate_PseudoChargeDensity(pSPARC);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCalculating b & b_ref took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();   
#endif

    // find atoms that have nonlocal influence the process domain (of psi-domain)
    GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc, pSPARC->DMVertices_dmcomm, 
                            pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nFinding nonlocal influencing atoms in psi-domain took %.3f ms\n",(t2-t1)*1000);
#endif

    if (nuChi0EigscommIndex == -1) return; // not take part in computation; Generate_PseudoChargeDensity has bcast to WORLD, have to put it here

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    
    // calculate nonlocal projectors in psi-domain
    if (pSPARC->isGammaPoint)
        CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                    pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
    else
        CalculateNonlocalProjectors_kpt(pSPARC, &pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                        pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);	                            
    
    if (pSPARC->SOC_Flag) {
        CalculateNonlocalProjectors_SOC(pSPARC, pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                                        pSPARC->DMVertices_dmcomm, pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
        CreateChiSOMatrix(pSPARC, pSPARC->nlocProj, pSPARC->Atom_Influence_nloc, 
                        pSPARC->bandcomm_index < 0 ? MPI_COMM_NULL : pSPARC->dmcomm);
    }
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCalculating nonlocal projectors in psi-domain took %.3f ms\n",(t2-t1)*1000);
    t1 = MPI_Wtime();   
#endif
    
    // find atoms that have nonlocal influence the process domain (of kptcomm_topo)
    GetInfluencingAtoms_nloc(pSPARC, &pSPARC->Atom_Influence_nloc_kptcomm, pSPARC->DMVertices_kptcomm, 
                            pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nFinding nonlocal influencing atoms in kptcomm_topo took %.3f ms\n",(t2-t1)*1000);
    t1 = MPI_Wtime();
#endif
    
    // calculate nonlocal projectors in kptcomm_topo
    if (pSPARC->isGammaPoint)
        CalculateNonlocalProjectors(pSPARC, &pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                    pSPARC->DMVertices_kptcomm, 
                                    pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
    else
        CalculateNonlocalProjectors_kpt(pSPARC, &pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                        pSPARC->DMVertices_kptcomm, 
                                        pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);								    
    
    if (pSPARC->SOC_Flag) {
        CalculateNonlocalProjectors_SOC(pSPARC, pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                                        pSPARC->DMVertices_kptcomm, 
                                        pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
        CreateChiSOMatrix(pSPARC, pSPARC->nlocProj_kptcomm, pSPARC->Atom_Influence_nloc_kptcomm, 
                        pSPARC->kptcomm_index < 0 ? MPI_COMM_NULL : pSPARC->kptcomm_topo);
    }
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("\nCalculating nonlocal projectors in kptcomm_topo took %.3f ms\n",(t2-t1)*1000);   
#endif
    
    // restore orbitals psi in every nuChi0Eigscomm
    restore_orbitals(pSPARC, nuChi0Eigscomm, nuChi0EigsBridgeComm, nuChi0EigscommIndex, rank0nuChi0EigscommInWorld, symk1, symk2, symk3, kPqSymList);
    
    restore_electronDensity(pSPARC, nuChi0Eigscomm, nuChi0EigsBridgeComm, nuChi0EigscommIndex, rank0nuChi0EigscommInWorld);
    
    restore_eigval_occ(pSPARC, nuChi0Eigscomm, nuChi0EigsBridgeComm, nuChi0EigscommIndex, Nkpts_sym, kPqSymList);
    
    int DMnd = pSPARC->Nd_d;
    int i;
    // solve the poisson equation for electrostatic potential, "phi"
    Calculate_elecstPotential(pSPARC);

    #ifdef DEBUG
    t1 = MPI_Wtime();
    #endif
    // calculate xc potential (LDA), "Vxc"
    Calculate_Vxc(pSPARC);
    pSPARC->countPotentialCalculate++;
	#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("rank = %d, XC calculation took %.3f ms\n", rank, (t2-t1)*1e3); 
    t1 = MPI_Wtime(); 
	#endif 
    
    // calculate Veff_loc_dmcomm_phi = phi + Vxc in "phi-domain"
    Calculate_Veff_loc_dmcomm_phi(pSPARC);
    
    // initialize mixing_hist_xk (and mixing_hist_xkm1)
    Update_mixing_hist_xk(pSPARC);
    
    // transfer Veff_loc from "phi-domain" to "psi-domain"
    for (i = 0; i < pSPARC->Nspden; i++)
        Transfer_Veff_loc_RPA(pSPARC, nuChi0Eigscomm, pSPARC->Veff_loc_dmcomm_phi + i*DMnd, pSPARC->Veff_loc_dmcomm + i*pSPARC->Nd_d_dmcomm);
    
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) {
        printf("rank = %d, Veff calculation and Bcast (non-blocking) took %.3f ms\n",rank,(t2-t1)*1e3); 
    }
	#endif
}

void restore_orbitals(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld,
     double *symk1, double *symk2, double *symk3, int **kPqSymList) {
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int flagPrintNuChi0Eigscomm = (nuChi0EigscommIndex == 0); // set which nuchi0Eigscomm will print output
#ifdef DEBUG
    if (flagPrintNuChi0Eigscomm && (rank == 0)) printf("restoring Kohn-Sham orbitals ... \n");
#endif

    int DMnd, DMndsp, size_k, len_tot;
#ifdef DEBUG
    double t1, t2;
#endif
    // Multiply a factor for a spinor wavefunction
    DMnd = pSPARC->Nd_d_dmcomm;
    DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    size_k = DMndsp * pSPARC->Nband_bandcomm;
    // notice that in processors not for orbital calculations len_tot = 0
    len_tot = size_k * pSPARC->Nkpts_kptcomm;

    if (pSPARC->isGammaPoint){
        // allocate memory in the very first relax/MD step
        pSPARC->Xorb = (double *)malloc( len_tot * sizeof(double) );
        pSPARC->Yorb = (double *)malloc( size_k * sizeof(double) );
        if (pSPARC->Xorb == NULL || pSPARC->Yorb == NULL) {
            printf("\nMemory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
#ifdef DEBUG            
        t1 = MPI_Wtime();
#endif
        if (nuChi0EigscommIndex == 0) { // read orbitals, then broadcast through nuChi0EigsBridgeCommIndex
            read_orbitals_distributed_gamma_RPA(pSPARC, nuChi0Eigscomm); // read orbitals of kpts
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb, len_tot, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
            }
        } else { // receiver of broadcast
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb, len_tot, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
            }
        }
        if (pSPARC->dmcomm != MPI_COMM_NULL) {
            int XorbIndex = 2;
            printf("I am rank %d in nuChi0Eigscomm %d, pSPARC->Xorb[%d] is %.6E\n", rank, nuChi0EigscommIndex, XorbIndex, pSPARC->Xorb[XorbIndex]);
        }
    } else {
        // allocate memory in the very first relax/MD step
        pSPARC->Xorb_kpt = (double _Complex *) malloc( len_tot * sizeof(double _Complex) );
        pSPARC->Yorb_kpt = (double _Complex *) malloc( size_k * sizeof(double _Complex) );
        if (pSPARC->Xorb_kpt == NULL || pSPARC->Yorb_kpt == NULL) {
            printf("\nMemory allocation failed!\n");
            exit(EXIT_FAILURE);
        }
#ifdef DEBUG            
        t1 = MPI_Wtime();
#endif                
        if (nuChi0EigscommIndex == 0) { // read orbitals, then broadcast through nuChi0EigsBridgeCommIndex
            read_orbitals_distributed_kpt_RPA(pSPARC, nuChi0Eigscomm, symk1, symk2, symk3, kPqSymList); // read orbitals of kpts
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb_kpt, len_tot, MPI_DOUBLE_COMPLEX, 0, nuChi0EigsBridgeComm);
            }
        } else { // receiver of broadcast
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Bcast(pSPARC->Xorb_kpt, len_tot, MPI_DOUBLE_COMPLEX, 0, nuChi0EigsBridgeComm);
            }
        }
        if (pSPARC->dmcomm != MPI_COMM_NULL) {
            int XorbIndex = 2;
            printf("I am rank %d in nuChi0Eigscomm %d, start k %d, |%d| - 1 in sym kpt list, pSPARC->Xorb[%d] is %.6E + %.6Ei\n", rank, nuChi0EigscommIndex,
                 pSPARC->kpt_start_indx, kPqSymList[pSPARC->kpt_start_indx][0], XorbIndex, creal(pSPARC->Xorb_kpt[XorbIndex]),  cimag(pSPARC->Xorb_kpt[XorbIndex]));
        }
    }        
#ifdef DEBUG
    t2 = MPI_Wtime();
    if(!rank) printf("Finished restoring orbitals. Time taken: %.3f ms\n",(t2-t1)*1e3);
#endif
}

void read_orbitals_distributed_gamma_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm) {
    if (pSPARC->dmcomm == MPI_COMM_NULL) return; // only dmcomm != NULL, the processor can join read; in this section, read only orbitals belonging to kpts inside symmetric kpt list
    int spin, band;
    int gridsizes[3]; // z, y, x
    gridsizes[0] = pSPARC->Nz; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nx;
    int localGridsizes[3]; // z, y, x
    localGridsizes[0] = pSPARC->Nz_d_dmcomm; localGridsizes[1] = pSPARC->Ny_d_dmcomm; localGridsizes[2] = pSPARC->Nx_d_dmcomm;
    int localGridStart[3]; // z, y, x
    localGridStart[0] = pSPARC->DMVertices_dmcomm[4]; localGridStart[1] = pSPARC->DMVertices_dmcomm[2]; localGridStart[2] = pSPARC->DMVertices_dmcomm[0];
    MPI_Datatype dataType = MPI_DOUBLE_COMPLEX;
    MPI_Datatype domainSubarray;
    MPI_Type_create_subarray(3, gridsizes, localGridsizes, localGridStart, MPI_ORDER_C, dataType, &domainSubarray);
    MPI_Type_commit(&domainSubarray);
    for (band = pSPARC->band_start_indx; band < pSPARC->band_end_indx + 1; band++) {
        for (spin = pSPARC->spin_start_indx; spin < pSPARC->spin_end_indx + 1; spin++) {
                read_orbitals_distributed_real_RPA(pSPARC, band, spin, domainSubarray, pSPARC->Xorb);
        }
    }
    MPI_Type_free(&domainSubarray);
}

void read_orbitals_distributed_real_RPA(SPARC_OBJ *pSPARC, int band, int spin, MPI_Datatype domainSubarray, double *readXorb) {
    MPI_Status status;
    MPI_File orbitalFh;
    char orbitalFileName[L_STRING];
    int localSpin = spin - pSPARC->spin_start_indx;
    int localBand = band - pSPARC->band_start_indx;
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int length = 0;
    while (pSPARC->filename[length] != '\0') {
        length++;
    }
    while ((length > -1) && (pSPARC->filename[length] != '/')) {
        length--; // find the working directory
    }
    char folderName[L_STRING];
    memcpy(folderName, pSPARC->filename, sizeof(char) * (length + 1));
    folderName[length + 1] = '\0';
    snprintf(orbitalFileName, L_STRING, "%sorbitals/band%d_spin%d.orbit", folderName, band, spin);
    printf("the name of the orbital file is %s, folder directory length %d\n", orbitalFileName, length);
    int openFile = MPI_File_open(pSPARC->dmcomm, orbitalFileName, MPI_MODE_RDWR, MPI_INFO_NULL, &orbitalFh); // cannot create a file
    if (openFile != MPI_SUCCESS) {
        printf("failed reading orbital %s!\n", orbitalFileName);
        exit(EXIT_FAILURE);
    }
    MPI_File_set_view(orbitalFh, 0, MPI_DOUBLE, domainSubarray, "native", MPI_INFO_NULL);
    MPI_File_read_all(orbitalFh, readXorb + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, &status);
    MPI_File_close(&orbitalFh);
}

void read_orbitals_distributed_kpt_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *symk1, double *symk2, double *symk3, int **kPqSymList) {
    if (pSPARC->dmcomm != MPI_COMM_NULL) { // only dmcomm != NULL, the processor can join read; in this section, read only orbitals belonging to kpts inside symmetric kpt list
        int spin, kpt, kptSym, band;
        int gridsizes[3]; // z, y, x
        gridsizes[0] = pSPARC->Nz; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nx;
        int localGridsizes[3]; // z, y, x
        localGridsizes[0] = pSPARC->Nz_d_dmcomm; localGridsizes[1] = pSPARC->Ny_d_dmcomm; localGridsizes[2] = pSPARC->Nx_d_dmcomm;
        int localGridStart[3]; // z, y, x
        localGridStart[0] = pSPARC->DMVertices_dmcomm[4]; localGridStart[1] = pSPARC->DMVertices_dmcomm[2]; localGridStart[2] = pSPARC->DMVertices_dmcomm[0];
        MPI_Datatype dataType = MPI_DOUBLE_COMPLEX;
        MPI_Datatype domainSubarray;
        MPI_Type_create_subarray(3, gridsizes, localGridsizes, localGridStart, MPI_ORDER_C, dataType, &domainSubarray);
        MPI_Type_commit(&domainSubarray);
        for (kpt = pSPARC->kpt_start_indx; kpt < pSPARC->kpt_end_indx + 1; kpt++) {
            if (kPqSymList[kpt][0] < 0) continue; // this kpt is not in symmetric kpt list
            kptSym = kPqSymList[kpt][0] - 1;
            for (band = pSPARC->band_start_indx; band < pSPARC->band_end_indx + 1; band++) {
                for (spin = pSPARC->spin_start_indx; spin < pSPARC->spin_end_indx + 1; spin++) {
                        read_orbitals_distributed_complex_RPA(pSPARC, 1, kpt, symk1[kptSym], symk2[kptSym], symk3[kptSym], band, spin, domainSubarray, pSPARC->Xorb_kpt);
                }
            }
        }
        MPI_Type_free(&domainSubarray);
    }
    MPI_Barrier(nuChi0Eigscomm); // prevent file-read conflict between kpt comms inside sym list and outside sym list
    if (pSPARC->dmcomm != MPI_COMM_NULL) { // only dmcomm != NULL, the processor can join read; in this section, read only orbitals belonging to kpts outside symmetric kpt list
        int spin, kpt, kptSym, band;
        int gridsizes[3]; // z, y, x
        gridsizes[0] = pSPARC->Nz; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nx;
        int localGridsizes[3]; // z, y, x
        localGridsizes[0] = pSPARC->Nz_d_dmcomm; localGridsizes[1] = pSPARC->Ny_d_dmcomm; localGridsizes[2] = pSPARC->Nx_d_dmcomm;
        int localGridStart[3]; // z, y, x
        localGridStart[0] = pSPARC->DMVertices_dmcomm[4]; localGridStart[1] = pSPARC->DMVertices_dmcomm[2]; localGridStart[2] = pSPARC->DMVertices_dmcomm[0];
        MPI_Datatype dataType = MPI_DOUBLE_COMPLEX;
        MPI_Datatype domainSubarray;
        MPI_Type_create_subarray(3, gridsizes, localGridsizes, localGridStart, MPI_ORDER_C, dataType, &domainSubarray);
        MPI_Type_commit(&domainSubarray);
        for (kpt = pSPARC->kpt_start_indx; kpt < pSPARC->kpt_end_indx + 1; kpt++) {
            if (kPqSymList[kpt][0] >= 0) continue; // this kpt is not in symmetric kpt list
            kptSym = -kPqSymList[kpt][0] - 1;
            for (band = pSPARC->band_start_indx; band < pSPARC->band_end_indx + 1; band++) {
                for (spin = pSPARC->spin_start_indx; spin < pSPARC->spin_end_indx + 1; spin++) {
                    read_orbitals_distributed_complex_RPA(pSPARC, 0, kpt, symk1[kptSym], symk2[kptSym], symk3[kptSym], band, spin, domainSubarray, pSPARC->Xorb_kpt);
                }
            }
        }
        MPI_Type_free(&domainSubarray);
    }
}


void read_orbitals_distributed_complex_RPA(SPARC_OBJ *pSPARC, int flagSymKpt, int kpt, double symk1, double symk2, double symk3, int band, int spin, MPI_Datatype domainSubarray, double _Complex *readXorb_kpt) {
    MPI_Status status;
    MPI_File orbitalFh;
    char orbitalFileName[L_STRING];
    int localSpin = spin - pSPARC->spin_start_indx;
    int localKpt = kpt - pSPARC->kpt_start_indx;
    int localBand = band - pSPARC->band_start_indx;
    int DMndsp = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_spincomm;
    int size_k = DMndsp * pSPARC->Nband_bandcomm;
    int length = 0;
    while (pSPARC->filename[length] != '\0') {
        length++;
    }
    while ((length > -1) && (pSPARC->filename[length] != '/')) {
        length--; // find the working directory
    }
    char folderName[L_STRING];
    memcpy(folderName, pSPARC->filename, sizeof(char) * (length + 1));
    folderName[length + 1] = '\0';
    snprintf(orbitalFileName, L_STRING, "%sorbitals/kpt_%.4f,%.4f,%.4f_band%d_spin%d.orbit", folderName, symk1, symk2, symk3, band, spin);
    printf("the name of the orbital file is %s, folder directory length %d\n", orbitalFileName, length);
    int openFile = MPI_File_open(pSPARC->dmcomm, orbitalFileName, MPI_MODE_RDWR, MPI_INFO_NULL, &orbitalFh); // cannot create a file
    if (openFile != MPI_SUCCESS) {
        printf("failed reading orbital %s!\n", orbitalFileName);
        exit(EXIT_FAILURE);
    }
    MPI_File_set_view(orbitalFh, 0, MPI_C_DOUBLE_COMPLEX, domainSubarray, "native", MPI_INFO_NULL);
    MPI_File_read_all(orbitalFh, readXorb_kpt + localKpt*size_k + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, MPI_C_DOUBLE_COMPLEX, &status);
    if (!flagSymKpt) { // this k-point is not in symmetric kpt list, so the orbital it just read is the conjugate of this kpt
        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
            *(readXorb_kpt + localKpt*size_k + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm + index) = conj(*(readXorb_kpt + localKpt*size_k + localBand*DMndsp + localSpin*pSPARC->Nd_d_dmcomm + index));
        }
    }
    MPI_File_close(&orbitalFh);
}

void restore_electronDensity(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld) {
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int flagPrintNuChi0Eigscomm = (nuChi0EigscommIndex == 0); // set which nuchi0Eigscomm will print output
#ifdef DEBUG
    if (flagPrintNuChi0Eigscomm && (rank == 0)) printf("restoring electron density ... \n");
#endif
    
    if (pSPARC->dmcomm_phi != MPI_COMM_NULL) {
        int phiRank;
        MPI_Comm_rank(pSPARC->dmcomm_phi, &phiRank);
        int DMnd = pSPARC->Nd_d;
        if (flagPrintNuChi0Eigscomm) { // 0th nuChi0Eigscomm read electron density
            // for 1st Relax step/ MDstep, set initial electron density
            // read initial density from file 
            char inputDensFnames[3][L_STRING+L_PSD];
            // set up input density filename
            if (phiRank == 0) {
                snprintf(inputDensFnames[0], L_STRING+L_PSD, "%s.dens", pSPARC->filename);
                snprintf(inputDensFnames[1], L_STRING+L_PSD, "%s.densUp", pSPARC->filename);
                snprintf(inputDensFnames[2], L_STRING+L_PSD, "%s.densDwn", pSPARC->filename);
            }
            int nFileToRead;
            if (pSPARC->Nspin == 1) { // spin-unpolarized or collinear spin
                nFileToRead = 1;
            } else { // spin-polarized
                nFileToRead = 3;
            }
            read_cube_and_dist_vec(
                pSPARC, inputDensFnames, pSPARC->electronDens, nFileToRead,
                pSPARC->DMVertices, pSPARC->dmcomm_phi
            );
            MPI_Bcast(pSPARC->electronDens, DMnd, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
        } else {  // other nuChi0Eigscomms receive electron density from 0th nuChi0Eigscomm
            MPI_Bcast(pSPARC->electronDens, DMnd, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
        }
        if (flagPrintNuChi0Eigscomm && (!rank)) {
            int densityIndex = 2;
            printf("I am rank %d in nuChi0Eigscomm %d; the %dth density is %9.6f\n", rank, nuChi0EigscommIndex, densityIndex, pSPARC->electronDens[densityIndex]);
        }
    }
}

void restore_eigval_occ(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int Nkpts_sym, int **kPqSymList) {
    // Nkpts_sym comes from pRPA. At here Nkpts_sym in pSPARC is equal to Nkpt
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int Ns = pSPARC->Nstates;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    int nspin = pSPARC->Nspin;
    int Nspin_spincomm = pSPARC->Nspin_spincomm;
    double *coordsKptsSym = (double*)calloc(sizeof(double), 3*Nkpts_sym);
    double *eigsKptsSym = (double*)calloc(sizeof(double), Ns * Nkpts_sym * nspin);
    double *occsKptsSym = (double*)calloc(sizeof(double), Ns * Nkpts_sym * nspin);
    if (nuChi0EigscommIndex == 0) {
        if (!rank) {
            char inputEigsFnames[L_STRING+L_PSD];
            snprintf(inputEigsFnames, L_STRING + 6, "%s.eigen", pSPARC->filename);
            read_eigval_occ(inputEigsFnames, pSPARC->Nspin, Nkpts_sym, Ns, coordsKptsSym, eigsKptsSym, occsKptsSym);
            MPI_Bcast(coordsKptsSym, 3*Nkpts_sym, MPI_DOUBLE, 0, nuChi0Eigscomm);
            MPI_Bcast(eigsKptsSym, Ns * Nkpts_sym * nspin, MPI_DOUBLE, 0, nuChi0Eigscomm);
            MPI_Bcast(occsKptsSym, Ns * Nkpts_sym * nspin, MPI_DOUBLE, 0, nuChi0Eigscomm);
        } else {
            MPI_Bcast(coordsKptsSym, 3*Nkpts_sym, MPI_DOUBLE, 0, nuChi0Eigscomm);
            MPI_Bcast(eigsKptsSym, Ns * Nkpts_sym * nspin, MPI_DOUBLE, 0, nuChi0Eigscomm);
            MPI_Bcast(occsKptsSym, Ns * Nkpts_sym * nspin, MPI_DOUBLE, 0, nuChi0Eigscomm);
        }
        if (pSPARC->bandcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) { // every processor in a kpt comm has ALL lambda and occ of the k-point!
        // consistent with the way SPARC saves the eigenvalues and occupations
            find_eigval_occ_spin_kpts(pSPARC, Nkpts_sym, coordsKptsSym, eigsKptsSym, occsKptsSym, kPqSymList);
            MPI_Bcast(pSPARC->lambda, Ns * Nkpts_kptcomm * Nspin_spincomm, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
            MPI_Bcast(pSPARC->occ, Ns * Nkpts_kptcomm * Nspin_spincomm, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
        }
    } else {
        if (pSPARC->bandcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
            MPI_Bcast(pSPARC->lambda, Ns * Nkpts_kptcomm * Nspin_spincomm, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
            MPI_Bcast(pSPARC->occ, Ns * Nkpts_kptcomm * Nspin_spincomm, MPI_DOUBLE, 0, nuChi0EigsBridgeComm);
        }
    }
    free(coordsKptsSym);
    free(eigsKptsSym);
    free(occsKptsSym);
    if ((pSPARC->bandcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL)) {
        printf("I am %d in nuChi0Eigscomm %d, my first spin %d, last kpt is %d, the second eig is %f, occ %f\n", rank, nuChi0EigscommIndex, pSPARC->spin_start_indx, pSPARC->kpt_end_indx,
            pSPARC->lambda[(Nkpts_kptcomm - 1)*Ns + (2 - 1)], pSPARC->occ[(Nkpts_kptcomm - 1)*Ns + (2 - 1)]);
    }
}

void read_eigval_occ(char *inputEigsFnames, int Nspin, int Nkpts_sym, int Ns, double *coordsKptsSym, double *eigsKptsSym, double *occsKptsSym) {
    FILE *eig_fp = fopen(inputEigsFnames, "r");
    if (eig_fp == NULL) {
        printf("Cannot open file \"%s\"\n", inputEigsFnames);
        exit(EXIT_FAILURE);
    }
    for (int kptSym = 0; kptSym < Nkpts_sym; kptSym++) {
        fscanf(eig_fp, "%*[^\n]\n");
        fscanf(eig_fp, "%*[^(](%lf,%lf,%lf)\n", &coordsKptsSym[kptSym*3], &coordsKptsSym[kptSym*3 + 1], &coordsKptsSym[kptSym*3 + 2]); // kred #1 = (-0.333333,0.250000,0.000000)
        fscanf(eig_fp, "%*[^\n]\n"); // weight = 0.333333
        fscanf(eig_fp, "%*[^\n]\n"); // n        eigval                 occ
        if (Nspin == 1) {
            for (int bandIndex = 0; bandIndex  < Ns; bandIndex++) {
                int theBand;
                fscanf(eig_fp, "%d %lf %lf", &theBand, &eigsKptsSym[kptSym*Ns + bandIndex], &occsKptsSym[kptSym*Ns + bandIndex]);
                fscanf(eig_fp, "%*[^\n]\n");
            }
        } else if (Nspin == 2) {
            for (int bandIndex = 0; bandIndex  < Ns; bandIndex++) {
                int theBand;
                fscanf(eig_fp, "%d %lf %lf %lf %lf", &theBand,
                     &eigsKptsSym[kptSym*Ns + bandIndex], &occsKptsSym[kptSym*Ns + bandIndex], &eigsKptsSym[Nkpts_sym*Ns + kptSym*Ns + bandIndex], &occsKptsSym[Nkpts_sym*Ns + kptSym*Ns + bandIndex]);
                fscanf(eig_fp, "%*[^\n]\n");
            }
        }
    }
    fclose(eig_fp);
}

void find_eigval_occ_spin_kpts(SPARC_OBJ *pSPARC, int Nkpts_sym, double *coordsKptsSym, double *eigsKptsSym, double *occsKptsSym, int **kPqSymList) {
    int Ns = pSPARC->Nstates;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    for (int spin = pSPARC->spin_start_indx; spin < pSPARC->spin_end_indx + 1; spin++) {
        int localSpin = spin - pSPARC->spin_start_indx;
        for (int kpt = pSPARC->kpt_start_indx; kpt < pSPARC->kpt_end_indx + 1; kpt++) {
            int kptSym, localKpt;
            localKpt = kpt - pSPARC->kpt_start_indx;
            if (kPqSymList[kpt][0] > 0) {
                kptSym = kPqSymList[kpt][0] - 1;
            } else {
                kptSym = -kPqSymList[kpt][0] - 1;
            }
            for (int bandIndex = 0; bandIndex < Ns; bandIndex++) { // every processor in a kpt comm has ALL lambda and occ of the k-point!
                // consistent with the way SPARC saves the eigenvalues and occupations
                pSPARC->lambda[localSpin*Nkpts_kptcomm*Ns + localKpt*Ns + bandIndex] = eigsKptsSym[spin*Nkpts_sym*Ns + kptSym*Ns + bandIndex];
                pSPARC->occ[localSpin*Nkpts_kptcomm*Ns + localKpt*Ns + bandIndex] = occsKptsSym[spin*Nkpts_sym*Ns + kptSym*Ns + bandIndex] / pSPARC->occfac; // the occ in .eigen file is multiplied by occfac
            }
        }
    }
}

/**
 * @brief   Transfer Veff_loc from phi-domain to psi-domain.
 *
 *          Use DD2DD (Domain Decomposition to Domain Decomposition) to 
 *          do the transmision between phi-domain and the dmcomm that 
 *          contains root process, and then broadcast to all dmcomms.
 */
void Transfer_Veff_loc_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *Veff_phi_domain, double *Veff_psi_domain) 
{
#ifdef DEBUG
    double t1, t2;
#endif
    
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
#ifdef DEBUG
    if (rank == 0) printf("Transmitting Veff_loc from phi-domain to psi-domain (LOCAL) ...\n");
#endif    
    //void DD2DD(SPARC_OBJ *pSPARC, int *gridsizes, int *sDMVert, double *sdata, int *rDMVert, double *rdata, 
    //       MPI_Comm send_comm, int *sdims, MPI_Comm recv_comm, int *rdims)
    int gridsizes[3], sdims[3], rdims[3];
    gridsizes[0] = pSPARC->Nx; gridsizes[1] = pSPARC->Ny; gridsizes[2] = pSPARC->Nz;
    sdims[0] = pSPARC->npNdx_phi; sdims[1] = pSPARC->npNdy_phi; sdims[2] = pSPARC->npNdz_phi;
    rdims[0] = pSPARC->npNdx; rdims[1] = pSPARC->npNdy; rdims[2] = pSPARC->npNdz;

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif
    D2D(&pSPARC->d2d_dmcomm_phi, &pSPARC->d2d_dmcomm, gridsizes, pSPARC->DMVertices, Veff_phi_domain, 
        pSPARC->DMVertices_dmcomm, Veff_psi_domain, pSPARC->dmcomm_phi, sdims, 
        (pSPARC->spincomm_index == 0 && pSPARC->kptcomm_index == 0 && pSPARC->bandcomm_index == 0) ? pSPARC->dmcomm : MPI_COMM_NULL, 
        rdims, nuChi0Eigscomm, sizeof(double));
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: D2D took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first kptcomms in each spincomm
    if (pSPARC->npspin > 1 && pSPARC->spincomm_index >= 0 && pSPARC->kptcomm_index == 0) {
        MPI_Bcast(Veff_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->spin_bridge_comm);
    }
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: bcast btw/ spincomms of 1st kptcomm took %.3f ms\n",(t2-t1)*1e3);
    t1 = MPI_Wtime();
#endif
    
    // Broadcast phi from the dmcomm that contain root process to all dmcomms of the first bandcomms in each kptcomm
    if (pSPARC->spincomm_index >= 0 && pSPARC->npkpt > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->bandcomm_index == 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(Veff_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->kpt_bridge_comm);
    }
    
#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: bcast btw/ kptcomms of 1st bandcomm took %.3f ms\n",(t2-t1)*1e3);
#endif

    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast below on Quartz cluster, this Barrier fixed the issue (why?)

#ifdef DEBUG
    t1 = MPI_Wtime();
#endif

    // Bcast phi from first bandcomm to all other bandcomms
    if (pSPARC->npband > 1 && pSPARC->kptcomm_index >= 0 && pSPARC->dmcomm != MPI_COMM_NULL) {
        MPI_Bcast(Veff_psi_domain, pSPARC->Nd_d_dmcomm, MPI_DOUBLE, 0, pSPARC->blacscomm);    
    }
    pSPARC->req_veff_loc = MPI_REQUEST_NULL;
    
    MPI_Barrier(pSPARC->blacscomm); // experienced severe slowdown of MPI_Bcast above on Quartz cluster, this Barrier fixed the issue (why?)

#ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("---Transfer Veff_loc: mpi_bcast (count = %d) to all bandcomms took %.3f ms\n",pSPARC->Nd_d_dmcomm,(t2-t1)*1e3);
#endif

    if ((pSPARC->ixc[2]) && (pSPARC->countPotentialCalculate > 1))
        Transfer_vxcMGGA3_phi_psi(pSPARC, pSPARC->vxcMGGA3, pSPARC->vxcMGGA3_loc_dmcomm); // only transfer the potential they are going to use
}
