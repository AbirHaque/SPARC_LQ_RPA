/**
 * @file    tools_RPA.c
 * @brief   This file saves simple functions for basic algebra operations.
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
#include <complex.h>
#include <mpi.h>

#include "tools_RPA.h"

void divide_complex_vectors(double _Complex *complexVecs, double *realPart, double *imagPart, int length) {
    for (int i = 0; i < length; i++) {
        realPart[i] = creal(complexVecs[i]);
        imagPart[i] = cimag(complexVecs[i]);
    }
}

// available only in conditions without domain parallelization
// if there is domain parallelization, then it is necessary to call pdgemm_ function in ScaLapack with blacs
// to make the distributed matrix multiplication
void matrix_multiplication(const double _Complex *M, int MsizeRow, int MsizeCol, const double _Complex *x, int numVecs, double _Complex *Mx) { // LHS*RHS
    int veclength = MsizeCol;
    int Mxlength = MsizeRow*numVecs;
    if (MsizeCol != veclength) {
        printf("Input matrix size error for multiplication. %d %d\n", MsizeCol, veclength);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < Mxlength; i++) {
        Mx[i] = 0.0 + 0.0*I;
    }
    int xindex = 0;
    for (int xcol = 0; xcol < numVecs; xcol++) {
        int Mindex = 0;
        for (int xrow = 0; xrow < veclength; xrow++) { // xrow, or Mcol
            int Mxindex = xcol*MsizeRow;
            double _Complex xentry = x[xindex]; // x(xrow, Mxcol)
            for (int Mrow = 0; Mrow < MsizeRow; Mrow++) {
                double _Complex Mentry = M[Mindex]; // M(Mrow, xrow)
                Mx[Mxindex] += Mentry * xentry; // Mx(Mrow, xcol)
                Mindex++;
                Mxindex++;
            }
            xindex++;
        }
    }
}

// available only in conditions without domain parallelization
void matrix_transpose(const double _Complex *M, int vecLength, int numVecs, double _Complex *MT) { 
    int Mindex = 0;
    for (int Mcol = 0; Mcol < numVecs; Mcol++) {
        int MTindex = Mcol;
        for (int Mrow = 0; Mrow < vecLength; Mrow++) {
            MT[MTindex] = M[Mindex];
            Mindex++;
            MTindex += numVecs;
        }
    }
}


/**
 * @brief   Calculate global sum of a vector among the given communicator. 
 */
void VectorSumComplex(const double _Complex *Vec, const int len, double _Complex *vec_sum, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    int k;
    double _Complex sum = 0.0;
    for (k = 0; k < len; k++)
        sum += Vec[k];
    if (comm != MPI_COMM_SELF) {
        MPI_Allreduce(&sum, vec_sum, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
    } else {
        *vec_sum = sum;
    }
}

/**
 * @brief   Calculate shift of a vector, x = x + c.
 */
void VectorShiftComplex(double _Complex *Vec, const int len, const double _Complex c, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    int k;
    for (k = 0; k < len; k++)
        Vec[k] += c;
}

void VectorScaleComplex(double _Complex *Vec, const int len, const double c, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    for (int k = 0; k < len; k++)
        Vec[k] *= c;
}