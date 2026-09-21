/**
 * @file    electrostatics_RPA.c
 * @brief   This file saves functions for multiplying f(\nabla^2), including multiplying sqrt(\nu) with vectors, and
 *          Sternheimer equation preconditioner
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "stddef.h"
#include "electrostatics.h"
#include "parallelization.h"
#include "gradVecRoutines.h"
#include "tools.h"
#include "linearSolver.h"
#include "lapVecRoutines.h"
#include "exactExchange.h"

#include "restoreElectronicGroundState.h"
#include "electrostatics_RPA.h"
#include "linearSolvers.h"
#include "tools_RPA.h"
#include "kroneckerLaplacian.h"

void collect_deltaRho_gamma(SPARC_OBJ *pSPARC, double *deltaRhos, int nuChi0EigsAmount, int printFlag, int nuChi0EigscommIndex) {
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    int DMnd = pSPARC->Nd_d_dmcomm;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (!flagNoDmcomm) {
        int dmcommRank;
        MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
        // sum over spin comm group
        if(pSPARC->npspin > 1) {        
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos, nuChi0EigsAmount*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->spin_bridge_comm);        
        }
        // sum over all k-point groups
        if (pSPARC->npkpt > 1) {            
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos, nuChi0EigsAmount*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
        // sum over all band groups 
        if (pSPARC->npband) {
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos, nuChi0EigsAmount*DMnd, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
        }
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0) && (!dmcommRank) && printFlag) {
            char deltaRhoFileName[100];
            snprintf(deltaRhoFileName, 100, "deltaRhos_nuChi0%d.txt", nuChi0EigscommIndex);
            FILE *outputDrhos = fopen(deltaRhoFileName, "w");
            if (outputDrhos ==  NULL) {
                printf("error printing delta rho s in test\n");
                exit(EXIT_FAILURE);
            } else {
                for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                    for (int index = 0; index < DMnd; index++) {
                        fprintf(outputDrhos, "%12.9f\n", deltaRhos[nuChi0EigIndex*DMnd + index]);
                    }
                    fprintf(outputDrhos, "\n");
                }
            }
            fclose(outputDrhos);
        }
    }
}

void collect_deltaRho_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt, int nuChi0EigsAmount, int printFlag, int nuChi0EigscommIndex) {
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    int DMnd = pSPARC->Nd_d_dmcomm;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    if (!flagNoDmcomm) {
        int dmcommRank;
        MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
        // sum over spin comm group
        if(pSPARC->npspin > 1) {        
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos_kpt, nuChi0EigsAmount*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->spin_bridge_comm);        
        }
        // sum over all k-point groups
        if (pSPARC->npkpt > 1) {            
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos_kpt, nuChi0EigsAmount*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->kpt_bridge_comm);
        }
        // sum over all band groups 
        if (pSPARC->npband) {
            MPI_Allreduce(MPI_IN_PLACE, deltaRhos_kpt, nuChi0EigsAmount*DMnd, MPI_DOUBLE_COMPLEX, MPI_SUM, pSPARC->blacscomm);
        }
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0) && (!dmcommRank) && printFlag) {
            char deltaRhoFileName[100];
            snprintf(deltaRhoFileName, 100, "deltaRhos_nuChi0%d.txt", nuChi0EigscommIndex);
            FILE *outputDrhos = fopen(deltaRhoFileName, "w");
            if (outputDrhos ==  NULL) {
                printf("error printing delta rho s in test\n");
                exit(EXIT_FAILURE);
            } else {
                for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                    for (int index = 0; index < DMnd; index++) {
                        fprintf(outputDrhos, "%12.9f %12.9f\n", creal(deltaRhos_kpt[nuChi0EigIndex*DMnd + index]), cimag(deltaRhos_kpt[nuChi0EigIndex*DMnd + index]));
                    }
                    fprintf(outputDrhos, "\n");
                }
            }
            fclose(outputDrhos);
        }
    }
}


void Calculate_sqrtNu_vecs_gamma(SPARC_OBJ *pSPARC, double *Vs, double *sqrtNuVs, int nuChi0EigsAmount, int printFlag, int flagNoDmcomm, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm) {
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    int Nd_d = pSPARC->Nd_d_dmcomm;
    if (flagNoDmcomm) {
        return;
    }
    
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmount; nuChi0EigsIndex++) {
    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
    #endif
        double deltaV_shift = 0.0;
        VectorSum (Vs + nuChi0EigsIndex*Nd_d, Nd_d, &deltaV_shift, pSPARC->dmcomm);
        if (pSPARC->BC == 2) {
            deltaV_shift /= (double)pSPARC->Nd;
            VectorShift(Vs + nuChi0EigsIndex*Nd_d, Nd_d, -deltaV_shift, pSPARC->dmcomm);
        } else { // minus has been added into the sqrt of the inverse of eigenvalues of Laplacian. (line 194, initialization_RPA.c)
            double sqrt4pi = sqrt(4.0*M_PI);
            for (int i = 0; i < Nd_d; i++) {
                Vs[nuChi0EigsIndex*Nd_d + i] = Vs[nuChi0EigsIndex*Nd_d + i] * sqrt4pi;
            }
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if(!globalRank) 
            printf("nuChi0Eigscomm %d, global rank %d, nuChi0EigsIndex %d, initial int_rhs = %18.14f, checking this took %.3f ms\n",
                nuChi0EigscommIndex, globalRank, nuChi0EigsIndex, deltaV_shift, (t2-t1)*1e3);
        t1 = MPI_Wtime();
    #endif    
        // no need to multiply 4pi on the rhs, it is included in pois_const 
        // pois_kron(pSPARC, deltaRhos + nuChi0EigsIndex*Nd_d, pSPARC->pois_const, 1, deltaVs + nuChi0EigsIndex*Nd_d);

        KRON_LAP* kron_lap = pSPARC->kron_lap_exx[0];
        if (pSPARC->BC == 2) {
            Lap_Kron(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx, kron_lap->Vy, kron_lap->Vz,
                    Vs + nuChi0EigsIndex*Nd_d, pSPARC->pois_const, sqrtNuVs + nuChi0EigsIndex*Nd_d);
        } else { // Dirichlet BC
            Lap_Kron(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx, kron_lap->Vy, kron_lap->Vz,
                Vs + nuChi0EigsIndex*Nd_d, kron_lap->inv_eig, sqrtNuVs + nuChi0EigsIndex*Nd_d);
        }
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (globalRank == 0) printf("Solving Poisson took %.3f ms\n", (t2-t1)*1e3);
    #endif

        // shift the electrostatic potential so that its integral is zero for periodic systems
        if (pSPARC->BC == 2) {
            deltaV_shift = 0.0;
            VectorSum  (sqrtNuVs + nuChi0EigsIndex*Nd_d, Nd_d, &deltaV_shift, pSPARC->dmcomm);
            deltaV_shift /= (double)pSPARC->Nd;
            VectorShift(sqrtNuVs + nuChi0EigsIndex*Nd_d, Nd_d, -deltaV_shift, pSPARC->dmcomm);
        }
    }
    
    if (printFlag) {
        if ((pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0)){
            int dmcommRank;
            MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
            if (dmcommRank == 0) {
                FILE *outputDVs = fopen("sqrtNuVs.txt", "w");
                if (outputDVs ==  NULL) {
                    printf("error printing sqrtNuVs in test\n");
                    exit(EXIT_FAILURE);
                } else {
                    for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                        for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                            fprintf(outputDVs, "%12.9f\n", sqrtNuVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]);
                        }
                        fprintf(outputDVs, "\n");
                    }
                }
                fclose(outputDVs);
            }
        }
    }
}


void Calculate_sqrtNu_vecs_kpt(SPARC_OBJ *pSPARC, double _Complex *Vs, double _Complex *sqrtNuVs, int qptIndex, int nuChi0EigsAmount, int printFlag, int flagNoDmcomm, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm) {
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    int globalRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    int Nd_d = pSPARC->Nd_d_dmcomm;
    if (flagNoDmcomm) {
        return;
    }

    if (printFlag && (globalRank == 0)) {
        int dmcommRank;
        MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
        if (dmcommRank == 0) {
            FILE *outputDVs = fopen("Vs.txt", "w");
            if (outputDVs ==  NULL) {
                printf("error printing Vs in test\n");
                exit(EXIT_FAILURE);
            } else {
                for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                    for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                        fprintf(outputDVs, "%12.9f+%12.9fi \n", creal(Vs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]), cimag(Vs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]));
                    }
                    fprintf(outputDVs, "\n");
                }
            }
            fclose(outputDVs);
        }
    }

    KRON_LAP* kron_lap = pSPARC->kron_lap_exx[qptIndex];
    for (int nuChi0EigsIndex = 0; nuChi0EigsIndex < nuChi0EigsAmount; nuChi0EigsIndex++) {
    #ifdef DEBUG
        double t1, t2;
        t1 = MPI_Wtime();
    #endif    
        // no need to multiply 4pi on the rhs, it is included in pois_const 

        Lap_Kron_complex(kron_lap->Nx, kron_lap->Ny, kron_lap->Nz, kron_lap->Vx_kpt, kron_lap->Vy_kpt, kron_lap->Vz_kpt,
                kron_lap->VyH_kpt, kron_lap->VzH_kpt, Vs + nuChi0EigsIndex*Nd_d, pSPARC->pois_const + qptIndex*Nd_d, sqrtNuVs + nuChi0EigsIndex*Nd_d);
    #ifdef DEBUG
        t2 = MPI_Wtime();
        if (globalRank == 0) printf("Solving Poisson took %.3f ms\n", (t2-t1)*1e3);
    #endif
    }
    
    if (printFlag && (globalRank == 0)) {
        int dmcommRank;
        MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
        if (dmcommRank == 0) {
            FILE *outputDVs = fopen("sqrtNuVs.txt", "w");
            if (outputDVs ==  NULL) {
                printf("error printing sqrtNuVs in test\n");
                exit(EXIT_FAILURE);
            } else {
                for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                    for (int index = 0; index < pSPARC->Nd_d_dmcomm; index++) {
                        fprintf(outputDVs, "%12.9f+%12.9fi \n", creal(sqrtNuVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]), cimag(sqrtNuVs[nuChi0EigIndex*pSPARC->Nd_d_dmcomm + index]));
                    }
                    fprintf(outputDVs, "\n");
                }
            }
            fclose(outputDVs);
        }
    }
}

