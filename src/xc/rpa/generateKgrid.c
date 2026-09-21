/**
 * @file    generateKgrid.c
 * @brief   This file contains the functions for calculating k-point and q-point grid. k-point grid is complete with shift;
 *          q-point grid is reduced by time-reverse symmetry without shift.
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
#include <mpi.h>
#include <limits.h>

#include "generateKgrid.h"

#define TEMP_TOL 1e-12

void transfer_kpoints(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA) {
 // move symmetric k-point grid from pSPARC to pRPA
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    memcpy(pRPA->kptWts, pSPARC->kptWts, pRPA->Nkpts_sym * sizeof(double));
    memcpy(pRPA->k1, pSPARC->k1, pRPA->Nkpts_sym * sizeof(double));
    memcpy(pRPA->k2, pSPARC->k2, pRPA->Nkpts_sym * sizeof(double));
    memcpy(pRPA->k3, pSPARC->k3, pRPA->Nkpts_sym * sizeof(double));
    #ifdef DEBUG
    if (!rank) printf("with symmetry reduction, Nkpts_sym = %d\n", pRPA->Nkpts_sym);
    for (int nk = 0; nk < pRPA->Nkpts_sym; nk++) {
        double tpiblx = 2 * M_PI / Lx;
        double tpibly = 2 * M_PI / Ly;
        double tpiblz = 2 * M_PI / Lz;
        if (!rank) printf("k1[%2d]: %8.4f, k2[%2d]: %8.4f, k3[%2d]: %8.4f, kptwt[%2d]: %.3f \n",
            nk,pRPA->k1[nk]/tpiblx,nk,pRPA->k2[nk]/tpibly,nk,pRPA->k3[nk]/tpiblz,nk,pRPA->kptWts[nk]);
    }
    #endif
}

void recalculate_kpoints(SPARC_OBJ *pSPARC) {
// to reset k-point grid as complete, without reduction
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double Lx = pSPARC->range_x;
    double Ly = pSPARC->range_y;
    double Lz = pSPARC->range_z;
    // calculate M-P grid similar to that in ABINIT
    int nk1_s = -floor((pSPARC->Kx - 1)/2);
    int nk1_e = nk1_s + pSPARC->Kx;
    int nk2_s = -floor((pSPARC->Ky - 1)/2);
    int nk2_e = nk2_s + pSPARC->Ky;
    int nk3_s = -floor((pSPARC->Kz - 1)/2);
    int nk3_e = nk3_s + pSPARC->Kz;
    int nk1, nk2, nk3;
    double k1_red, k2_red, k3_red;
    double k1, k2, k3;
    int nk;
    int nkpts = 0;

    for (nk1 = nk1_s; nk1 < nk1_e; nk1++) {
        for (nk2 = nk2_s; nk2 < nk2_e; nk2++) {
            for (nk3 = nk3_s; nk3 < nk3_e; nk3++) {
                k1_red = nk1 * 1.0/pSPARC->Kx;
                k2_red = nk2 * 1.0/pSPARC->Ky;
                k3_red = nk3 * 1.0/pSPARC->Kz;
                k1_red = fmod(k1_red + pSPARC->kptshift[0] / pSPARC->Kx + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k2_red = fmod(k2_red + pSPARC->kptshift[1] / pSPARC->Ky + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k3_red = fmod(k3_red + pSPARC->kptshift[2] / pSPARC->Kz + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k1 = k1_red * 2.0 * M_PI / Lx;
                k2 = k2_red * 2.0 * M_PI / Ly;
                k3 = k3_red * 2.0 * M_PI / Lz;
                // do not reduce k-point
                pSPARC->k1[nkpts] = k1;
                pSPARC->k2[nkpts] = k2;
                pSPARC->k3[nkpts] = k3;
                pSPARC->kptWts[nkpts]= 1.0;
                nkpts++;
            }
        }
    }
    pSPARC->Nkpts_sym = nkpts; // restore number of k points 
    #ifdef DEBUG
    if (!rank) printf("without symmetry reduction, Nkpts = %d\n", nkpts);
    for (nk = 0; nk < pSPARC->Nkpts_sym; nk++) {
        double tpiblx = 2 * M_PI / Lx;
        double tpibly = 2 * M_PI / Ly;
        double tpiblz = 2 * M_PI / Lz;
        if (!rank) printf("k1[%2d]: %8.4f, k2[%2d]: %8.4f, k3[%2d]: %8.4f, kptwt[%2d]: %.3f \n",
            nk,pSPARC->k1[nk]/tpiblx,nk,pSPARC->k2[nk]/tpibly,nk,pSPARC->k3[nk]/tpiblz,nk,pSPARC->kptWts[nk]);
    }
    #endif
}

int set_qpoints(double *qptWts, double *q1, double *q2, double *q3, int Kx, int Ky, int Kz, double Lx, double Ly, double Lz) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double sumx = 2.0 * M_PI / Lx;
    double sumy = 2.0 * M_PI / Ly;
    double sumz = 2.0 * M_PI / Lz;
    // calculate M-P grid similar to that in ABINIT
    int nk1_s = -floor((Kx - 1)/2);
    int nk1_e = nk1_s + Kx;
    int nk2_s = -floor((Ky - 1)/2);
    int nk2_e = nk2_s + Ky;
    int nk3_s = -floor((Kz - 1)/2);
    int nk3_e = nk3_s + Kz;
    int nk1, nk2, nk3;
    double k1_red, k2_red, k3_red;
    double k1, k2, k3;
    int nk, flag;
    int Nqpts_sym = 0;

    for (nk1 = nk1_s; nk1 < nk1_e; nk1++) {
        for (nk2 = nk2_s; nk2 < nk2_e; nk2++) {
            for (nk3 = nk3_s; nk3 < nk3_e; nk3++) {
                k1_red = nk1 * 1.0/Kx;
                k2_red = nk2 * 1.0/Ky;
                k3_red = nk3 * 1.0/Kz;
                k1_red = fmod(k1_red + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k2_red = fmod(k2_red + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k3_red = fmod(k3_red + 0.5 - TEMP_TOL, 1.0) - 0.5 + TEMP_TOL;
                k1 = k1_red * 2.0 * M_PI / Lx;
                k2 = k2_red * 2.0 * M_PI / Ly;
                k3 = k3_red * 2.0 * M_PI / Lz;
                flag = 1;
                for (nk = 0; nk < Nqpts_sym; nk++) { // to remove symmetric k-points
                    if (   (fabs(k1 + q1[nk]) < TEMP_TOL || fabs(k1 + q1[nk] - sumx) < TEMP_TOL) 
                        && (fabs(k2 + q2[nk]) < TEMP_TOL || fabs(k2 + q2[nk] - sumy) < TEMP_TOL)
                        && (fabs(k3 + q3[nk]) < TEMP_TOL || fabs(k3 + q3[nk] - sumz) < TEMP_TOL) ) {
                        flag = 0;
                        break;
                    }
                }
                if (flag) {
                    q1[Nqpts_sym] = k1;
                    q2[Nqpts_sym] = k2;
                    q3[Nqpts_sym] = k3;
                    qptWts[Nqpts_sym]= 1.0;
                    Nqpts_sym++;
                } else {
                    qptWts[nk] = 2.0;
                }
            }
        }
    }
    if (!rank) printf("After symmetry reduction, Nqpts_sym = %d\n", Nqpts_sym);
    #ifdef DEBUG
    double tpiblx = 2 * M_PI / Lx;
    double tpibly = 2 * M_PI / Ly;
    double tpiblz = 2 * M_PI / Lz;
    for (nk = 0; nk < Nqpts_sym; nk++) {
        if (!rank) printf("q1[%2d]: %8.4f, q2[%2d]: %8.4f, q3[%2d]: %8.4f, qptwt[%2d]: %.3f \n",
            nk,q1[nk]/tpiblx,nk,q2[nk]/tpibly,nk,q3[nk]/tpiblz,nk,qptWts[nk]);
    }
    #endif
    return Nqpts_sym;
}

void set_kPq_kMq_lists(int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym, int Nkpts, double *k1, double *k2, double *k3, 
    int Nqpts_sym, double *q1, double *q2, double *q3, double Lx, double Ly, double Lz, int **kPqSymList, int **kPqList, int **kMqList) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // kPqSymList: 2-D int array, abs(kPqSymList[k index complete table][q index]) - 1 = k+q index sym table
    // abs(kPqSymList[k index complete table][0]) - 1 = k index sym table
    // if kPqSymList[nk][nq] < 0, the correcponding k-point in sym table is symmetric of the nk th k-point in complete tabke. The \psis should be conjugate
    // the reason for +1 in indices of kPqSymList is to distinguish +0th and -0th index in sym table
    for (int nk = 0; nk < Nkpts; nk++) {
        kPqSymList[nk][0] = find_kpt_sym_index(k1[nk], k2[nk], k3[nk], Nkpts_sym, k1sym, k2sym, k3sym, Lx, Ly, Lz);
        for (int nq = 0; nq < Nqpts_sym; nq++) {
            kPqSymList[nk][nq + 1] = find_kpt_sym_index(k1[nk] + q1[nq], k2[nk] + q2[nq], k3[nk] + q3[nq], Nkpts_sym, k1sym, k2sym, k3sym, Lx, Ly, Lz);
        }
    }
    #ifdef DEBUG
    if (!rank) {
        printf("abs value of indices in kPqSymList are +1 larger than index of its corresponding kpt in kpt sym table\n");
        for (int nk = 0; nk < Nkpts; nk++) {
            printf("kPqSymList[%d]: %d", nk, kPqSymList[nk][0]);
            for (int nq = 0; nq < Nqpts_sym; nq++) {
                printf(" %d", kPqSymList[nk][nq + 1]);
            }
            printf("\n");
        }
    }
    #endif
    int kPqSym = INT_MAX;
    for (int nk = 0; nk < Nkpts; nk++) {
        for (int nq = 0; nq < Nqpts_sym; nq++){
            kPqSym = kPqSymList[nk][nq + 1];
            for (int possiblek = 0; possiblek < Nkpts; possiblek++) {
                if (kPqSymList[possiblek][0] == kPqSym) {
                    kPqList[nk][nq] = possiblek;
                    kMqList[possiblek][nq] = nk;
                    break;
                }
            }
        }
    }
    #ifdef DEBUG
    if (!rank) {
        printf("kPqList and kMqList are mappings between complete k-points.\n");
        for (int nk = 0; nk < Nkpts; nk++) {
            printf("kPqList[%d]: ", nk);
            for (int nq = 0; nq < Nqpts_sym; nq++) {
                printf(" %d", kPqList[nk][nq]);
            }
            printf("\n");
        }
        for (int nk = 0; nk < Nkpts; nk++) {
            printf("kMqList[%d]: ", nk);
            for (int nq = 0; nq < Nqpts_sym; nq++) {
                printf(" %d", kMqList[nk][nq]);
            }
            printf("\n");
        }
    }
    #endif
}

int find_kpt_sym_index(double k1Coord, double k2Coord, double k3Coord, int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym, double Lx, double Ly, double Lz) {
    double tpiblx = 2 * M_PI / Lx;
    double tpibly = 2 * M_PI / Ly;
    double tpiblz = 2 * M_PI / Lz;
    if (k1Coord > tpiblx/2.0) k1Coord -= tpiblx;
    else if (k1Coord < -tpiblx/2.0) k1Coord += tpiblx;
    if (k2Coord > tpibly/2.0) k2Coord -= tpibly;
    else if (k2Coord < -tpibly/2.0) k2Coord += tpibly;
    if (k3Coord > tpiblz/2.0) k3Coord -= tpiblz;
    else if (k3Coord < -tpiblz/2.0) k3Coord += tpiblz;
    int kptSymIndex = find_kpt_sym_1to1(k1Coord, k2Coord, k3Coord, Nkpts_sym, k1sym, k2sym, k3sym);
    if (kptSymIndex == INT_MAX) {
        if (fabs(k1Coord - tpiblx/2.0) < TEMP_TOL)
            k1Coord = - tpiblx/2.0;
        else if (fabs(k1Coord + tpiblx/2.0) < TEMP_TOL)
            k1Coord = tpiblx/2.0;
        kptSymIndex = find_kpt_sym_1to1(k1Coord, k2Coord, k3Coord, Nkpts_sym, k1sym, k2sym, k3sym);
    }
    if (kptSymIndex == INT_MAX) {
        if (fabs(k2Coord - tpibly/2.0) < TEMP_TOL)
            k2Coord = - tpibly/2.0;
        else if (fabs(k2Coord + tpibly/2.0) < TEMP_TOL)
            k2Coord = tpibly/2.0;
        kptSymIndex = find_kpt_sym_1to1(k1Coord, k2Coord, k3Coord, Nkpts_sym, k1sym, k2sym, k3sym);
    }
    if (kptSymIndex == INT_MAX) {
        if (fabs(k3Coord - tpiblz/2.0) < TEMP_TOL)
            k3Coord = - tpiblz/2.0;
        else if (fabs(k3Coord + tpiblz/2.0) < TEMP_TOL)
            k3Coord = tpiblz/2.0;
        kptSymIndex = find_kpt_sym_1to1(k1Coord, k2Coord, k3Coord, Nkpts_sym, k1sym, k2sym, k3sym);
    }
    if (kptSymIndex == INT_MAX) {
        printf("Cannot find the k+q point %8.4f %8.4f %8.4f in kpt sym list!\n", k1Coord / tpiblx, k2Coord / tpibly, k3Coord / tpiblz);
        exit(EXIT_FAILURE);
    }
    return kptSymIndex;
}

int find_kpt_sym_1to1(double k1Coord, double k2Coord, double k3Coord, int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym) {
    int kptSymIndex = INT_MAX;
    double diff1, diff2;
    for (int symIndex = 0; symIndex < Nkpts_sym; symIndex++) {
        diff1 = sqrt((k1Coord - k1sym[symIndex])*(k1Coord - k1sym[symIndex]) + (k2Coord - k2sym[symIndex])*(k2Coord - k2sym[symIndex]) + (k3Coord - k3sym[symIndex])*(k3Coord - k3sym[symIndex]));
        if (diff1 < TEMP_TOL) {
            kptSymIndex = symIndex + 1;
            break;
        }
        diff2 = sqrt((k1Coord + k1sym[symIndex])*(k1Coord + k1sym[symIndex]) + (k2Coord + k2sym[symIndex])*(k2Coord + k2sym[symIndex]) + (k3Coord + k3sym[symIndex])*(k3Coord + k3sym[symIndex]));
        if (diff2 < TEMP_TOL) {
            kptSymIndex = -(symIndex + 1);
            break;
        }
    }
    return kptSymIndex;
}