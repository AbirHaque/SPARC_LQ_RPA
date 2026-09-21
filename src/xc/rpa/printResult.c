/**
 * @file    printResult.c
 * @brief   This file contains the functions for printing results.
 *
 * @authors Boqin Zhang <bzhang376@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "printResult.h"

void print_result(RPA_OBJ *pRPA, int nAtoms) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (((!pRPA->nuChi0EigscommIndex) && (!rank))||(pRPA->flagSQ && rank==0)) {
        double Erpa = 0.0;
        FILE *output_fp = fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"Energy terms in every (qpt, omega) pair (Ha/atom)\n");
        for (int qptIndex = 0; qptIndex < pRPA->Nqpts_sym; qptIndex++) {
            fprintf(output_fp,"q-point %d\n", qptIndex + 1);
            for (int omegaIndex = 0; omegaIndex < pRPA->Nomega; omegaIndex++) {
                double ErpaTerm = pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex];
                fprintf(output_fp,"omega %d: %.8E, ", omegaIndex + 1, ErpaTerm / (double)nAtoms);
                Erpa += ErpaTerm;
                if (omegaIndex % 3 == 2) fprintf(output_fp,"\n");
            }
            fprintf(output_fp,"\n");
        }
        fprintf(output_fp,"Total RPA correlation energy:\n");
        fprintf(output_fp,"%.8E (Ha), %.8E (Ha/atom)\n", Erpa, Erpa / (double)nAtoms);
        fclose(output_fp);
    }
}

void print_eigs_nuchi0(RPA_OBJ *pRPA, int qptIndex, int omegaIndex) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank) {
        FILE *output_eig = fopen(pRPA->filename_outEig,"a");
        fprintf(output_eig, "qptIndex %d, omegaIndex %d\n", qptIndex + 1, omegaIndex + 1);
        int remainder = pRPA->nuChi0Neig % 100;
        int totalLine = pRPA->nuChi0Neig / 100 + (remainder > 0 ? 1 : 0);
        int index = 0;
        for (int line = 0; line < totalLine; line++) {
            int localLength;
            if ((line == totalLine - 1) && (remainder)) {
                localLength = remainder;
            } else {
                localLength = 100;
            }
            for (int localIndex = 0; localIndex < localLength; localIndex++) {
                fprintf(output_eig, "%.3E ", pRPA->RRnuChi0Eigs[index]);
                index++;
            }
            fprintf(output_eig, "\n");
        }
        fprintf(output_eig, "\n");
        fclose(output_eig);
    }
}