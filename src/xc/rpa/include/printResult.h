#ifndef PRINTRPA
#define PRINTRPA

#include "main_RPA.h"

/**
 * @brief print the Erpa terms for every (qpt, omega) pair
 *
 * @param pRPA pointer to RPA_OBJ
 * @param nAtoms number of atoms in the system
 */
void print_result(RPA_OBJ *pRPA, int nAtoms);

/**
 * @brief print eigenvalues of every \Tilde{\chi}(qpt, omega) operator
 *
 * @param pRPA pointer to RPA_OBJ
 * @param qptIndex the index of qpt of the current nu chi0 operator
 * @param omegaIndex the index of omega of the current nu chi0 operator
 */
void print_eigs_nuchi0(RPA_OBJ *pRPA, int qptIndex, int omegaIndex);

#endif