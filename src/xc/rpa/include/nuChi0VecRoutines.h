#ifndef NUCHI0VECROUTINES
#define NUCHI0VECROUTINES

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief   Calculate the product of the target operator \tilde{\chi} = sqrt(\nu)*\chi^0(\omega)*sqrt(\nu) and trial vectors dVs
 *          
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param DVs (input) input vectors to multiply \tilde{\chi}, in dmcomm, not in dmcomm_phi
 * @param nuChi0DVs (output) product of \tilde{\chi} operator and input vectors
 * @param nuChi0EigsAmount amount of solved eigenvalues in this nuChi0Eigscomm, at here it is the amount of DV vectors
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param nIter the time of subspace iteration (power method)
 * @param printFlag flag to print mid variables
 */
void nuChi0_mult_vectors_gamma(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int omegaIndex, double *DVs, double *nuChi0DVs, int nuChi0EigsAmount, int flagNoDmcomm, int nIter, int printFlag);

/**
 * @brief   Calculate the product of the target operator \tilde{\chi^0} = sqrt(\nu)*\chi^0(\bold{q}, \omega)*sqrt(\nu) and trial vectors dVs
 *          
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param qptIndex the index of the q-point
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param DVs_kpt (input) input vectors to multiply \tilde{\chi}, in dmcomm, not in dmcomm_phi
 * @param nuChi0DVs_kpt (output) product of \tilde{\chi} operator and input vectors
 * @param nuChi0EigsAmount amount of solved eigenvalues in this nuChi0Eigscomm, at here it is the amount of DV vectors
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param nIter the time of subspace iteration (power method)
 * @param printFlag flag to print mid variables
 */
void nuChi0_mult_vectors_kpt(SPARC_OBJ* pSPARC, RPA_OBJ* pRPA, int qptIndex, int omegaIndex, double _Complex *DVs_kpt, double _Complex *nuChi0DVs_kpt, int nuChi0EigsAmount, int flagNoDmcomm, int nIter, int printFlag);
#endif