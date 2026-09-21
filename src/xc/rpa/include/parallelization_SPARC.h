#ifndef PARALLEL_SPARC
#define PARALLEL_SPARC

#include "isddft.h"

/**
 * @brief setup the communicators in SPARC object. Reminder: every trial vector communicator (nuChi0Eigscomm) has its own
 *        4-levels parallelization framework (spin, kpt, band, domain) with a set of communicators in SPARC object, saving
 *        all information about the previous K-S calculation.
 * @param pSPARC    the pointer that points to SPARC_OBJ type structure SPARC.
 * @param nuChi0Eigscomm the trial vector communicator (nuChi0Eigscomm) of this processor
 * @param nuChi0EigscommIndex the global index of the nuChi0Eigscomm to which this processor belongs
 * @param rank0nuChi0EigscommInWorld global rank of the 0th processor of the nu chi0 eigs communicator
*/
void Setup_Comms_SPARC(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

#endif