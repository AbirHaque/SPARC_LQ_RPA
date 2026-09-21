#ifndef TOOLSRPA
#define TOOLSRPA

void divide_complex_vectors(double _Complex *complexVecs, double *realPart, double *imagPart, int length);

void matrix_transpose(const double _Complex *M, int vecLength, int numVecs, double _Complex *MT);

void matrix_multiplication(const double _Complex *M, int MsizeRow, int MsizeCol, const double _Complex *x, int numVecs, double _Complex *Mx);

void VectorSumComplex(const double _Complex *Vec, const int len, double _Complex *vec_sum, MPI_Comm comm);

void VectorShiftComplex(double _Complex *Vec, const int len, const double _Complex c, MPI_Comm comm);

void VectorScaleComplex(double _Complex *Vec, const int len, const double c, MPI_Comm comm);

#endif