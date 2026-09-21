#ifndef COLLECTORBIT
#define COLLECTORBIT

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief Collect all psis, epsilons and occupations in different bandcomms, which are used for composing initial guess of Sternheimer eq.s
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 */
void collect_allXorb_allLambdas_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

/**
 * @brief Collect all psis, epsilons and occupations in different bandcomms, which are used for composing initial guess of Sternheimer eq.s
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 */
void collect_allXorb_allLambdas_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

/**
 * @brief Send the collected orbitals to the kpt comm handling k-q point, and receive the orbitals from the kpt comm handling k+q point.
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param qptIndex the index of qpt of the current nu chi0 operator
 */
void send_recv_allXorb_allLambdas_kPq(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex);

#endif