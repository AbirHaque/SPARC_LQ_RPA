#ifndef CHEBFILTERRPA
#define CHEBFILTERRPA

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief Initialize the Delta V vectors to make CheFSI. The initialized vectors are pointed by pRPA->deltaVs_phi
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 */
void initialize_deltaVs(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

/**
 * @brief Make CheFSI on every nu chi0(qpt, omega) operator to get its RPA energy term
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param qptIndex the index of qpt of the current nu chi0 operator
 * @param omegaIndex the index of omega of the current nu chi0 operator
 */
void subspace_iteration_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex);

/**
 * @brief Make a subspace iteration: project operator, eigensolver and vector update
 *          
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param rank rank of the processor in nuChi0 comm
 * @param nuChi0EigscommIndex index of nuChi0 comm which the processor belongs to
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param nIter the time of subspace iteration (power method)
 */
void project_tildeChi_general_eigProblem(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int omegaIndex, int flagNoDmcomm, int nIter);

/**
 * @brief Make a subspace iteration with k-point: project operator, eigensolver and vector update
 *          
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param rank rank of the processor in nuChi0 comm
 * @param nuChi0EigscommIndex index of nuChi0 comm which the processor belongs to
 * @param qptIndex the index of qpt of the current nu chi0 operator
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param nIter the time of subspace iteration (power method)
 */
void project_tildeChi_general_eigProblem_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int qptIndex, int omegaIndex, int flagNoDmcomm, int nIter);

/**
 * @brief Compute the RPA energy term of the nu chi0(qpt, omega) operator from its eigenvalues
 *
 * @param RRnuChi0Eigs eigenvalues of the nu chi0(qpt, omega) operator
 * @param nuChi0Neig amount of solved eigenvalues
 * @param omegaMesh01 meshs in interval 0~1, mapped from 0~infty, Gauss-Legendre integral 
 * @param qptOmegaWeight the weight of the current nu chi0(qpt, omega), the weight of qpt multiplying the weight of omega
 */
double compute_ErpaTerm(double *RRnuChi0Eigs, int nuChi0Neig, double omegaMesh01, double qptOmegaWeight);

#ifdef DEBUG
/**
 * @brief initialize timing counter for recording the wall time distribution for an (qpt, omega)
 *          
 * @param pRPA pointer to RPA_OBJ
 */
void initialize_timing_counter(RPA_OBJ *pRPA);

/**
 * @brief output the wall time distribution for an (qpt, omega)
 *          
 * @param pRPA pointer to RPA_OBJ
 * @param qptIndex the index of qpt of the current nu chi0 operator
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 */
void finalize_timing_counter(RPA_OBJ *pRPA, int qptIndex, int omegaIndex, int flagNoDmcomm);
#endif

#endif