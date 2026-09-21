#ifndef INITIALIZATION_RPA
#define INITIALIZATION_RPA

#include "isddft.h"

/**
 * @brief   Initializes parameters and objects of SPARC_OBJ, but stopped before setting communicators.
 *          Communicators in pSPARC, such as spin_comm, kpt_comm, band_comm..., rely on nuChi0Eigscomm
 *          in pRPA, so setting communicators and other jobs behind it are done after initialization
 *          of pRPA
 *
 * @param pSPARC   (out) The pointer that points to SPARC_OBJ type structure SPARC.
 * @param argc  The number of arguments that are passed to the program.
 * @param argv  The array of strings representing command line arguments.
 */
void Initialize_SPARC_before_SetComm(SPARC_OBJ *pSPARC, int argc, char *argv[]);

/**
 * @brief   Initializes parameters and objects of SPARC_OBJ, continue since setting communicators.
 *          Communicators in pSPARC, such as spin_comm, kpt_comm, band_comm..., rely on nuChi0Eigscomm
 *          in pRPA, so setting communicators and other jobs behind it are done after initialization
 *          of pRPA
 *
 * @param pSPARC   (out) The pointer that points to SPARC_OBJ type structure SPARC.
 * @param nuChi0Eigscomm the communicator saving some deltaV vectors (eigvectors of nu chi0), every nuChi0Eigscomm has a complete pSPARC saving all information about the previous K-S DFT calculation
 * @param nuChi0EigscommIndex the global index of the nuChi0Eigscomm to which this processor belongs
 * @param rank0nuChi0EigscommInWorld the global rank of the root of (my, this processor) nuChi0Eigscomm
 */
void Initialize_SPARC_SetComm_after(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

#endif