#ifndef CHEFSIRPASQ
#define CHEFSIRPASQ

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief Make CheFSI on every nu^0.5 chi0 nu^0.5 (qpt, omega) operator to get its RPA energy term by SQ
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param qptIndex the index of qpt of the current nu chi0 operator
 * @param omegaIndex the index of omega of the current nu chi0 operator
 */
void subspace_iteration_RPASQ(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex);

/**
 * @brief Make a subspace iteration: project operator and vector update, without eigensolver
 *          
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param rank rank of the processor in nuChi0 comm
 * @param nuChi0EigscommIndex index of nuChi0 comm which the processor belongs to
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param nIter the time of subspace iteration (power method)
 */
void project_tildeChi_update_deltaV(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int omegaIndex, int flagNoDmcomm, int nIter);

/**
 * @brief Make a subspace iteration: project operator and vector update, without eigensolver
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
void project_tildeChi_update_deltaV_kpt(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int rank, int nuChi0EigscommIndex, int qptIndex, int omegaIndex, int flagNoDmcomm, int nIter);

/**
 * @brief Estimate the trace of Tr(f(H)) by Spectrum quadrature
 * @param pRPA pointer to RPA_OBJ
 * @param HpSQ projected operator Hp = V'XV distributed in 1D topology in HpDiagComm
 * @param desc_HpSQ descripter of Hp operator in HpDiagComm
 * @param HpDiagStartIndex the start index of the diagonal entries to be solved by this HpDiagComm
 * @param HpDiagEndIndex the end index of the diagonal entries to be solved by this HpDiagComm
 * @param qptOmegaWeight w(qpt) * w(omega)
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param HpDiagComm HpDiag communicator to solve designated diagonal entries in ln(1 - Hp) + Hp
 * @param SQtimeRecorder array recording the time spent on every step of SQ
 * @param rankHpDiagComm rank of the processor in its HpDiagComm
 * @param HpPrintFlag flag to print mid variables
 */
double SQ_trace_estimator(RPA_OBJ *pRPA, double *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex,
    double qptOmegaWeight, int omegaIndex, MPI_Comm HpDiagComm, double *SQtimeRecorder, int rankHpDiagComm, int HpPrintFlag);

/**
 * @brief Estimate the trace of Tr(f(H)) by Spectrum quadrature
 * @param pRPA pointer to RPA_OBJ
 * @param HpSQ projected operator Hp = V'XV distributed in 1D topology in HpDiagComm
 * @param desc_HpSQ descripter of Hp operator in HpDiagComm
 * @param HpDiagStartIndex the start index of the diagonal entries to be solved by this HpDiagComm
 * @param HpDiagEndIndex the end index of the diagonal entries to be solved by this HpDiagComm
 * @param qptOmegaWeight w(qpt) * w(omega)
 * @param omegaIndex the index of omega of the current \tilde{\chi} operator
 * @param HpDiagComm HpDiag communicator to solve designated diagonal entries in ln(1 - Hp) + Hp
 * @param SQtimeRecorder array recording the time spent on every step of SQ
 * @param rankHpDiagComm rank of the processor in its HpDiagComm
 * @param HpPrintFlag flag to print mid variables
 */
double SQ_trace_estimator_kpt(RPA_OBJ *pRPA, double _Complex *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex,
    double qptOmegaWeight, int omegaIndex, MPI_Comm HpDiagComm, double *SQtimeRecorder, int rankHpDiagComm, int HpPrintFlag);

#endif