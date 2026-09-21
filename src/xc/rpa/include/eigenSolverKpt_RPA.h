#ifndef EIGKPTRPA
#define EIGKPTRPA

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief Multiply the filtered vectors Y^T by itself Y
 *
 * @param pRPA pointer to RPA_OBJ
 * @param dmcomm the psi domain communicator in pSPARC, saving filtered vectors Y
 * @param Y (input) the vectors to compute Y^T*Y
 * @param DMnd the length of vector y (a vector in Y) saved in the current processor, y is distributed in psi domain
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param Y_BLCYC (output) if pblas is used, then Y_BLCYC saves the row-wise distributed Y vectors
 * @param Mp (output) the matrix Mp = Y^T*Y, it is in row-wise partition form, controlled by descripter pRPA->desc_Mp_BLCYC
 * @param printFlag flag to print mid variables
 */
void YT_multiply_Y_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm, double _Complex *Y, int DMnd, int Nspinor_eig, int flagNoDmcomm, double _Complex *Y_BLCYC, double _Complex *Mp, int printFlag);

void Y_orth_permute_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, double _Complex *Ys_BLCYC, double _Complex *Mp, double _Complex *Ys_orthed, int flagNoDmcomm, int printFlag);

/**
 * @brief Project operator nu chi0 into space spanned by Y, Y^T * nu chi0 * Y
 *
 * @param pRPA pointer to RPA_OBJ
 * @param dmcomm the psi domain communicator in pSPARC, saving filtered vectors Y
 * @param Y_BLCYC (input) the vectors to compute Y^T*oparator*Y, but it should be in block cyclix form. The transform should be done in function YT_multiply_Y_gamma
 * @param HY (input) the vectors operator*Y to compute Y^T*oparator*Y
 * @param DMnd the length of vector y (a vector in Y) saved in the current processor, y is distributed in psi domain
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param Hp (output) the matrix Hp = Y^T*operator*Y, it is in row-wise partition form, comtrolled by descripter pRPA->desc_Hp_BLCYC
 * @param outputHY_BLCYC (output) the block cyclic distributed operator*Y, to be used later
 * @param printFlag flag to print mid variables
 */
void project_YT_nuChi0_Y_kpt(RPA_OBJ* pRPA, MPI_Comm dmcomm, double _Complex *Y_BLCYC, double _Complex *HY, int DMnd, int Nspinor_eig, int flagNoDmcomm, double _Complex *Hp, double _Complex *outputHY_BLCYC, int printFlag);

/**
 * @brief Solve the eigenpairs of (Y^T * operator * Y). It is a generalized eigenproblem (Y is not orthogonailized).
 *        Hp*Q = Mp*Q*Lambda
 *
 * @param pRPA pointer to RPA_OBJ
 * @param blacsComm blacsComm in pRPA (not pSPARC!), connecting processors of all nuChi0Eigscomms with equal rank
 * @param Hp (input) = Y^T * operator * Y
 * @param Mp (input) = Y^T * Y
 * @param nr the number of rows of Hp and Mp assigned in this processor
 * @param nc the number of columns of Hp and Mp assigned in this processor
 * @param descHp descripter of Hp
 * @param descMp descripter of Mp
 * @param eigValues (output) eigenvalues of the generalized eigenproblem
 * @param Q (output) eigenvectors of the generalized eigenproblem, it is in row-wise partition form, controlled by descripter pRPA->desc_Q_BLCYC
 * @param nrQ the number of rows of Q assigned in this processor
 * @param ncQ the number of columns Q assigned in this processor
 * @param descQ descripter of Q
 * @param blksz block size for distributing matrix Hp, Mp and Q, used if it is solved in parallel
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param printFlag flag to print mid variables
 */
void generalized_eigenproblem_solver_kpt(RPA_OBJ* pRPA, MPI_Comm blacsComm, 
    double _Complex *Hp, double _Complex *Mp, int nr, int nc, int *descHp, int *descMp,
    double *eigValues, double _Complex *Q, int nrQ, int ncQ, int *descQ,
    int blksz, int flagNoDmcomm, int printFlag);

/**
 * @brief Restore eigenvectors Q from space spanned by Y back to real space, Y*Q
 * @param pSPARC pointer to SPARC_OBJ (just for printing mid variables)
 * @param pRPA pointer to RPA_OBJ
 * @param dmcomm the phi domain in pSPARC, saving filtered vectors Y
 * @param DMnd the length of vector y (a vector in Y) saved in the current processor, y is distributed in psi domain
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param nuChi0Y_BLCYC (input) the vectors to be rotated, at here they are sqrt(nu) chi0 sqrt(nu) timing Ys, distributed in block cyclix
 * @param deltaVs (output) nuChi0Ys*Q = sqrt(nu) chi0 sqrt(nu) * deltaVs(new), used in the next subspace iteration
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param printFlag flag to print mid variables
 */
void subspace_rotation_update_deltaVs_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, MPI_Comm dmcomm, int DMnd, int Nspinor_eig, 
    double _Complex *nuChi0Y_BLCYC, double _Complex *deltaVs, int flagNoDmcomm, int printFlag);

#endif