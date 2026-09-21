#ifndef TESTHX
#define TESTHX

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief Test the validity of eigenpairs of Hamiltonian and the routine multiplying nu chi0 operator by vectors
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 */
void test_Hx_nuChi0(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

/**
 * @brief Test the validity of eigenpairs of Hamiltonian saved in every nuChi0Eigscomm
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param testHxAccuracy (out) the accuracy of H*psi - epsilon*psi
 */
void test_Hx(SPARC_OBJ *pSPARC, double *testHxAccuracy);

void test_eigSolver(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

#endif