#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "restoreElectronicGroundState.h"
#include "test_Hx_nuChi0_eigSolver.h"
#include "collectOrbitals.h"
#include "electrostatics_RPA.h"
#include "nuChi0VecRoutines.h"
#include "tools_RPA.h"
#include "eigenSolverGamma_RPA.h"
#include "eigenSolverKpt_RPA.h"
#include "sternheimerEquation.h"
#include "sternheimerEquationKpt.h"

void test_Hx_nuChi0(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    MPI_Comm nuChi0Eigscomm = pRPA->nuChi0Eigscomm;
    int rank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);

    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);

    if (!flagNoDmcomm) {
        double *testHxAccuracy = (double *)calloc(sizeof(double), pSPARC->Nkpts_kptcomm * pSPARC->Nspin_spincomm * pSPARC->Nband_bandcomm);
        test_Hx(pSPARC, testHxAccuracy);
        if ((nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->Nband_bandcomm > 0)) {
            printf("rank %d in nuChi0Eigscomm %d, the relative error epsilon*psi - H*psi of the bands %d %d are %.6E %.6E\n", rank, nuChi0EigscommIndex,
                   pSPARC->band_start_indx, pSPARC->band_start_indx + 1, testHxAccuracy[0], testHxAccuracy[1]);
        }
        free(testHxAccuracy);
    }

    if (nuChi0EigscommIndex == 0) { // this test is done in only one nuChi0Eigscomm pRPA->npnuChi0Neig - 1
        int printFlag = 1;
        int qptIndex = 1;
        int omegaIndex = pRPA->Nomega - 1; // 0;
        int nuChi0EigsAmount = pRPA->nNuChi0Eigscomm;// 1;
        if (qptIndex > pRPA->Nqpts_sym - 1)
            qptIndex = pRPA->Nqpts_sym - 1;
        if (omegaIndex > pRPA->Nomega - 1)
            omegaIndex = pRPA->Nomega - 1;
        if (pRPA->flagCOCGinitial && (!flagNoDmcomm)) {
            if (pSPARC->isGammaPoint) {
                collect_allXorb_allLambdas_gamma(pSPARC, pRPA);
            } else {
                collect_allXorb_allLambdas_kpt(pSPARC, pRPA);
                send_recv_allXorb_allLambdas_kPq(pSPARC, pRPA, qptIndex);
            }
        }
        int Nd_d_dmcomm = pSPARC->Nd_d_dmcomm;
        if (pSPARC->isGammaPoint) {
            double t1 = MPI_Wtime();
            double *sqrtNuDVs = pRPA->sprtNuDeltaVs;
            Calculate_sqrtNu_vecs_gamma(pSPARC, pRPA->deltaVs, sqrtNuDVs, nuChi0EigsAmount, printFlag, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
            double t3 = MPI_Wtime();
            if (!flagNoDmcomm) {
                sternheimer_eq_gamma(pSPARC, pRPA, omegaIndex, nuChi0EigsAmount, sqrtNuDVs, printFlag);
            }
            double t4 = MPI_Wtime();
            collect_deltaRho_gamma(pSPARC, pRPA->deltaRhos, nuChi0EigsAmount, printFlag, pRPA->nuChi0Eigscomm);
            Calculate_sqrtNu_vecs_gamma(pSPARC, pRPA->deltaRhos, pRPA->Ys, nuChi0EigsAmount, printFlag, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
            double t2 = MPI_Wtime();
            if (!rank) {
                FILE *outputFile = fopen(pRPA->timeRecordname, "a");
                fprintf(outputFile, "omega %d, sternheimer time %.3f, total Multiplication time is %.3f\n", omegaIndex, (t4 - t3)*1e3, (t2 - t1)*1e3);
                fclose(outputFile);
            }
            if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->spincomm_index == 0) && (pSPARC->bandcomm_index == 0) && (pSPARC->dmcomm != MPI_COMM_NULL)) { // print all \Delta V vectors of the last nuChi0Eigscomm.
                // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
                // only one processor print
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *output1stDV = fopen("dVs_tildeChidVs_test.txt", "w");
                    if (output1stDV ==  NULL) {
                        printf("error printing delta Vs in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        fprintf(output1stDV, "dVs\n");
                        for (int index = 0; index < Nd_d_dmcomm; index++) {
                            for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                                fprintf(output1stDV, "%12.9f ", pRPA->deltaVs[nuChi0EigIndex*Nd_d_dmcomm + index]);
                            }
                            fprintf(output1stDV, "\n");
                        }
                        fprintf(output1stDV, "tildeChidVs\n");
                        for (int index = 0; index < Nd_d_dmcomm; index++) {
                            for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                                fprintf(output1stDV, "%12.9f ", pRPA->Ys[nuChi0EigIndex*Nd_d_dmcomm + index]);
                            }
                            fprintf(output1stDV, "\n");
                        }
                    }
                    fclose(output1stDV);
                }
            }
        } else {
            if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->spincomm_index == 0) && (pSPARC->kptcomm_index == 0) && (pSPARC->bandcomm_index == 0) && (pSPARC->dmcomm != MPI_COMM_NULL)) { // print the first \Delta V vector of the last nuChi0Eigscomm.
                // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
                // only one processor print
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *output1stDV = fopen("deltaVs_kpt.txt", "w");
                    if (output1stDV ==  NULL) {
                        printf("error printing delta Vs in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                            for (int index = 0; index < Nd_d_dmcomm; index++) {
                                fprintf(output1stDV, "%12.9f %12.9f\n", creal(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]), cimag(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]));
                            }
                            fprintf(output1stDV, "\n");
                        }
                    }
                    fclose(output1stDV);
                }
            }
            double t1 = MPI_Wtime();
            double _Complex *sprtNuDVs_kpt = pRPA->sprtNuDeltaVs_kpt;
            Calculate_sqrtNu_vecs_kpt(pSPARC, pRPA->deltaVs_kpt, sprtNuDVs_kpt, qptIndex, nuChi0EigsAmount, printFlag, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
            double t3 = MPI_Wtime();
            if (!flagNoDmcomm) {
                sternheimer_eq_kpt(pSPARC, pRPA, qptIndex, omegaIndex, nuChi0EigsAmount, sprtNuDVs_kpt, printFlag);
            }
            double t4 = MPI_Wtime();
            collect_deltaRho_kpt(pSPARC, pRPA->deltaRhos_kpt, nuChi0EigsAmount, printFlag, pRPA->nuChi0Eigscomm);
            Calculate_sqrtNu_vecs_kpt(pSPARC, pRPA->deltaRhos_kpt, pRPA->Ys_kpt, qptIndex, nuChi0EigsAmount, 0, flagNoDmcomm, pRPA->nuChi0EigscommIndex, pRPA->nuChi0Eigscomm);
            double t2 = MPI_Wtime();
            if (!rank) {
                FILE *outputFile = fopen(pRPA->timeRecordname, "a");
                fprintf(outputFile, "omega %d, sternheimer time %.3f, total Multiplication time is %.3f\n", omegaIndex, (t4 - t3)*1e3, (t2 - t1)*1e3);
                fclose(outputFile);
            }
            if ((pRPA->nuChi0EigscommIndex == pRPA->npnuChi0Neig - 1) && (pSPARC->spincomm_index == 0) && (pSPARC->bandcomm_index == 0) && (pSPARC->dmcomm != MPI_COMM_NULL)) { // print all \Delta V vectors of the last nuChi0Eigscomm.
                // the code is only for cases without domain parallelization. In the case with domain parallelization, it needs to be modified by parallel output
                // only one processor print
                int dmcommRank;
                MPI_Comm_rank(pSPARC->dmcomm, &dmcommRank);
                if (dmcommRank == 0) {
                    FILE *output1stDV = fopen("dVs_tildeChidVs_test.txt", "w");
                    if (output1stDV ==  NULL) {
                        printf("error printing delta Vs in test\n");
                        exit(EXIT_FAILURE);
                    } else {
                        fprintf(output1stDV, "dVs\n");
                        for (int index = 0; index < Nd_d_dmcomm; index++) {
                            for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                                fprintf(output1stDV, "%12.9f + %12.9fi, ", creal(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]), cimag(pRPA->deltaVs_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]));
                            }
                            fprintf(output1stDV, "\n");
                        }
                        fprintf(output1stDV, "tildeChidVs\n");
                        for (int index = 0; index < Nd_d_dmcomm; index++) {
                            for (int nuChi0EigIndex = 0; nuChi0EigIndex < nuChi0EigsAmount; nuChi0EigIndex++) {
                                fprintf(output1stDV, "%12.9f + %12.9fi, ", creal(pRPA->Ys_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]), cimag(pRPA->Ys_kpt[nuChi0EigIndex*Nd_d_dmcomm + index]));
                            }
                            fprintf(output1stDV, "\n");
                        }
                    }
                    fclose(output1stDV);
                }
            }
        }
    }
}

void test_Hx(SPARC_OBJ *pSPARC, double *testHxAccuracy)
{
    int DMnd = pSPARC->Nd_d_dmcomm;
    int DMndsp = DMnd * pSPARC->Nspinor_spincomm;
    int *DMVertices = pSPARC->DMVertices_dmcomm;
    int ncol = pSPARC->Nband_bandcomm;
    int Nkpts_kptcomm = pSPARC->Nkpts_kptcomm;
    MPI_Comm comm = pSPARC->dmcomm;
    if (pSPARC->isGammaPoint) { // follow the sequence in function Calculate_elecDens, divide gamma-point, k-point first
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) { // follow the sequence in function eigSolve_CheFSI, divide spin
            int sg = pSPARC->spin_start_indx + spn_i;
            Hamiltonian_vectors_mult(
                pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
                pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb + spn_i * DMnd, DMndsp, pSPARC->Yorb + spn_i * DMnd, DMndsp, spn_i, comm);
            for (int bandIndex = 0; bandIndex < ncol; bandIndex++) { // verify the correctness of psi
                double *psi = pSPARC->Xorb + bandIndex * DMndsp + spn_i * DMnd;
                double *Hpsi = pSPARC->Yorb + bandIndex * DMndsp + spn_i * DMnd;
                double eigValue = pSPARC->lambda[spn_i * ncol + bandIndex];
                for (int i = 0; i < DMnd; i++)
                {
                    testHxAccuracy[spn_i * ncol + bandIndex] += (*(psi + i) * eigValue - *(Hpsi + i)) * (*(psi + i) * eigValue - *(Hpsi + i));
                }
            }
        }
    }
    else {
        int size_k = DMndsp * pSPARC->Nband_bandcomm;
        for (int spn_i = 0; spn_i < pSPARC->Nspin_spincomm; spn_i++) { // follow the sequence in function eigSolve_CheFSI_kpt
            int sg = pSPARC->spin_start_indx + spn_i;
            for (int kpt = 0; kpt < Nkpts_kptcomm; kpt++) {
                Hamiltonian_vectors_mult_kpt(
                    pSPARC, DMnd, DMVertices, pSPARC->Veff_loc_dmcomm + sg * pSPARC->Nd_d_dmcomm,
                    pSPARC->Atom_Influence_nloc, pSPARC->nlocProj, ncol, 0, pSPARC->Xorb_kpt + kpt * size_k + spn_i * DMnd, DMndsp, pSPARC->Yorb_kpt + spn_i * DMnd, DMndsp, spn_i, kpt, comm);
                for (int bandIndex = 0; bandIndex < ncol; bandIndex++) { // verify the correctness of psi
                    double _Complex *psi = pSPARC->Xorb_kpt + kpt * size_k + bandIndex * DMndsp + spn_i * DMnd;
                    double _Complex *Hpsi = pSPARC->Yorb_kpt + bandIndex * DMndsp + spn_i * DMnd;
                    double eigValue = pSPARC->lambda[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex];
                    for (int i = 0; i < DMnd; i++) {
                        testHxAccuracy[spn_i * Nkpts_kptcomm * ncol + kpt * ncol + bandIndex] += creal(conj(*(psi + i) * eigValue - *(Hpsi + i)) * (*(psi + i) * eigValue - *(Hpsi + i)));
                    }
                }
            }
        }
    }
}

void test_eigSolver(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
    int nuChi0EigscommIndex = pRPA->nuChi0EigscommIndex;
    if (nuChi0EigscommIndex == -1)
        return;
    int rank, size;
    MPI_Comm_rank(pRPA->nuChi0Eigscomm, &rank);
    MPI_Comm_size(pRPA->nuChi0Eigscomm, &size);
    int printFlag = 1;
    int flagNoDmcomm = (pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    int lengthY = pSPARC->Nd_d_dmcomm*pRPA->nNuChi0Eigscomm;
    if (rank == size - 1) {
        if (pSPARC->isGammaPoint) {
            double *midVector_BLCYC, *deltaVs_BLCYC, *Ys_BLCYC;
            if (pRPA->npnuChi0Neig > 1) {
                midVector_BLCYC = (double *)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double));
                deltaVs_BLCYC = midVector_BLCYC;
            } else {
                deltaVs_BLCYC = pRPA->Ys;
            }
            YT_multiply_Y_gamma(pRPA, pSPARC->dmcomm, pRPA->deltaVs, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, deltaVs_BLCYC, pRPA->Mp, printFlag);
            for (int vector = 0; vector < pRPA->nNuChi0Eigscomm; vector++) {
                for (int i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
                    pRPA->Ys[vector*pSPARC->Nd_d_dmcomm + i] = -pRPA->deltaVs[vector*pSPARC->Nd_d_dmcomm + i] / (double)(i + 1); // to replace nuChi0_mult_vectors_gamma 
                }
            }
            
            if (!pRPA->nuChi0EigsBridgeCommIndex) {
                if (pRPA->npnuChi0Neig > 1) {
                    Ys_BLCYC = midVector_BLCYC;
                } else {
                    Ys_BLCYC = pRPA->Ys;
                }
                project_YT_nuChi0_Y_gamma(pRPA, pSPARC->dmcomm, deltaVs_BLCYC, pRPA->Ys, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, pRPA->Hp, Ys_BLCYC, printFlag);
                
                generalized_eigenproblem_solver_gamma(pRPA, pRPA->nuChi0BlacsComm, 
                    pRPA->Hp, pRPA->Mp, pRPA->nr_Hp_BLCYC, pRPA->nc_Hp_BLCYC, pRPA->desc_Hp_BLCYC, pRPA->desc_Mp_BLCYC, 
                    pRPA->RRnuChi0Eigs, pRPA->Q, pRPA->nr_Q_BLCYC, pRPA->nc_Q_BLCYC, pRPA->desc_Q_BLCYC,
                    pSPARC->eig_paral_blksz, flagNoDmcomm, printFlag);
                int DMndspe = pSPARC->Nd_d_dmcomm * pSPARC->Nspinor_eig;

                subspace_rotation_update_deltaVs(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, 
                    Ys_BLCYC, pRPA->deltaVs, flagNoDmcomm, 0);

                    for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
                    double vec2norm;
                    Vector2Norm(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
                    VectorScale(pRPA->deltaVs + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
                }

                MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0BlacsComm);
            }

            MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
            MPI_Bcast(pRPA->deltaVs, pSPARC->Nd * pRPA->nNuChi0Eigscomm, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);

            if (pRPA->npnuChi0Neig > 1) {
                free(midVector_BLCYC);
            }
        } else {
            memcpy(pRPA->Ys_kpt, pRPA->deltaVs_kpt, sizeof(double _Complex)*lengthY); // to replace chebyshev filtering
            double _Complex *midVector_BLCYC, *deltaVs_kptBLCYC, *Ys_kptBLCYC;
            if (pRPA->npnuChi0Neig > 1) {
                midVector_BLCYC = (double _Complex*)malloc(pRPA->nr_orb_BLCYC * pRPA->nc_orb_BLCYC * sizeof(double _Complex));
                deltaVs_kptBLCYC = midVector_BLCYC;
            } else {
                deltaVs_kptBLCYC = pRPA->Ys_kpt;
            }
            YT_multiply_Y_kpt(pRPA, pSPARC->dmcomm, pRPA->deltaVs_kpt, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, deltaVs_kptBLCYC, pRPA->Mp_kpt, 0);
            for (int vector = 0; vector < pRPA->nNuChi0Eigscomm; vector++) {
                for (int i = 0; i < pSPARC->Nd_d_dmcomm; i++) {
                    pRPA->Ys_kpt[vector*pSPARC->Nd_d_dmcomm + i] = -pRPA->deltaVs_kpt[vector*pSPARC->Nd_d_dmcomm + i] / (double _Complex)(i + 1); // to replace nuChi0_mult_vectors_gamma 
                }
            }

            if (!pRPA->nuChi0EigsBridgeCommIndex) {
                if (pRPA->npnuChi0Neig > 1) {
                    Ys_kptBLCYC = midVector_BLCYC;
                } else {
                    Ys_kptBLCYC = pRPA->Ys_kpt;
                }
                project_YT_nuChi0_Y_kpt(pRPA, pSPARC->dmcomm, deltaVs_kptBLCYC, pRPA->Ys_kpt, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, flagNoDmcomm, pRPA->Hp_kpt, Ys_kptBLCYC, 0);
                
                generalized_eigenproblem_solver_kpt(pRPA, pRPA->nuChi0BlacsComm, 
                    pRPA->Hp_kpt, pRPA->Mp_kpt, pRPA->nr_Hp_BLCYC, pRPA->nc_Hp_BLCYC, pRPA->desc_Hp_BLCYC, pRPA->desc_Mp_BLCYC, 
                    pRPA->RRnuChi0Eigs, pRPA->Q_kpt, pRPA->nr_Q_BLCYC, pRPA->nc_Q_BLCYC, pRPA->desc_Q_BLCYC,
                    pSPARC->eig_paral_blksz, flagNoDmcomm, 0); // pSPARC->eig_paral_blksz

                subspace_rotation_update_deltaVs_kpt(pSPARC, pRPA, pSPARC->dmcomm, pSPARC->Nd_d_dmcomm, pSPARC->Nspinor_eig, 
                    Ys_kptBLCYC, pRPA->deltaVs_kpt, flagNoDmcomm, 0);

                for (int i = 0; i < pRPA->nNuChi0Eigscomm; i++) {
                    double vec2norm;
                    Vector2Norm_complex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, &vec2norm, pSPARC->dmcomm);
                    VectorScaleComplex(pRPA->deltaVs_kpt + i*pSPARC->Nd_d_dmcomm, pSPARC->Nd_d_dmcomm, 1.0/vec2norm, pSPARC->dmcomm); // unify the length of \Delta V
                }

                MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0BlacsComm);
            }
            MPI_Bcast(pRPA->RRnuChi0Eigs, pRPA->nuChi0Neig, MPI_DOUBLE, 0, pRPA->nuChi0Eigscomm);
            MPI_Bcast(pRPA->deltaVs_kpt, pSPARC->Nd * pRPA->nNuChi0Eigscomm, MPI_DOUBLE_COMPLEX, 0, pRPA->nuChi0Eigscomm);

            if (pRPA->npnuChi0Neig > 1) {
                free(midVector_BLCYC);
            }
        }
    }
}