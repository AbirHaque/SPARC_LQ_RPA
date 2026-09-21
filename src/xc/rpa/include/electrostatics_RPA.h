#ifndef ELECRPA
#define ELECRPA

#include "main_RPA.h"
#include "electrostatics_RPA.h"

/**
 * @brief Collect all delta rhos in different bandcomms to get delta rhos = chi0*deltaVs, these delta rho vectors are real vectors
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param deltaRhos delta rho vectors in different bandcomms
 * @param nuChi0EigsAmount the amount of deltaV vectors (eigvectors of nu chi0) saved in this nuChi0Eigscomm
 * @param printFlag flag to print mid variables
 * @param nuChi0EigscommIndex the index of communicator distributed deltaV vectors, every nuChi0Eigscomm has a complete pSPARC saving all information about the previous K-S DFT calculation
 */
void collect_deltaRho_gamma(SPARC_OBJ *pSPARC, double *deltaRhos, int nuChi0EigsAmount, int printFlag, int nuChi0EigscommIndex);

/**
 * @brief Collect all delta rhos in different bandcomms to get delta rhos = chi0*deltaVs, these delta rho vectors are complex vectors
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param deltaRhos_kpt delta rho vectors in different bandcomms
 * @param nuChi0EigsAmount the amount of deltaV vectors (eigvectors of nu chi0) saved in this nuChi0Eigscomm
 * @param printFlag flag to print mid variables
 * @param nuChi0EigscommIndex the index of communicator distributed deltaV vectors, every nuChi0Eigscomm has a complete pSPARC saving all information about the previous K-S DFT calculation
 */
void collect_deltaRho_kpt(SPARC_OBJ *pSPARC, double _Complex *deltaRhos_kpt, int nuChi0EigsAmount, int printFlag, int nuChi0EigscommIndex);

/**
 * @brief Compute the product of sqrt(nu) * deltaRhos. nu is Coulumb operator. It is done by Kronecker product
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param Vs (input) delta rho vectors in different bandcomms
 * @param sqrtNuVs (output) deltaV vectors,  which is the product of sqrt(Nu) and delta rho
 * @param nuChi0EigsAmount the amount of deltaV vectors (eigvectors of nu chi0) saved in this nuChi0Eigscomm
 * @param printFlag flag to print mid variables
 * @param nuChi0EigscommIndex the global index of the nuChi0Eigscomm to which this processor belongs
 * @param nuChi0Eigscomm  the trial vector communicator (nuChi0Eigscomm) of this processor, every nuChi0Eigscomm has a complete set of parallelization framework saving all information about the previous K-S DFT calculation
 */
void Calculate_sqrtNu_vecs_gamma(SPARC_OBJ *pSPARC, double *Vs, double *sqrtNuVs, int nuChi0EigsAmount, int printFlag, int flagNoDmcomm, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm);

/**
 * @brief Compute the product of sqrt(nu) * deltaRhos. nu is Coulumb operator. It is done by Kronecker product
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param Vs (input) delta rho vectors in different bandcomms
 * @param sqrtNuVs (output) deltaV vectors,  which is the product of sqrt(Nu) and delta rho
 * @param qptIndex the index of qpt being computed, marking the type of symmetry
 * @param nuChi0EigsAmount the amount of deltaV vectors (eigvectors of nu chi0) saved in this nuChi0Eigscomm
 * @param printFlag flag to print mid variables
 * @param nuChi0EigscommIndex the global index of the nuChi0Eigscomm to which this processor belongs
 * @param nuChi0Eigscomm  the trial vector communicator (nuChi0Eigscomm) of this processor, every nuChi0Eigscomm has a complete set of parallelization framework saving all information about the previous K-S DFT calculation
 */
void Calculate_sqrtNu_vecs_kpt(SPARC_OBJ *pSPARC, double _Complex *Vs, double _Complex *sqrtNuVs, int qptIndex, int nuChi0EigsAmount, int printFlag, int flagNoDmcomm, int nuChi0EigscommIndex, MPI_Comm nuChi0Eigscomm);

#endif