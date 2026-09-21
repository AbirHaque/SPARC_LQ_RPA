#ifndef GROUNDRPA 
#define GROUNDRPA 

#include "isddft.h"

/**
 * @brief restore all needed information and settings from the previous K-S calculation in the SPARC object
 *        K-S calculation has time-reverse symmetry (symmetrized k-points), but RPA needs to use complete k-point list.
 *        So necessary to find the mapping from the coord of k+q in complete k-point list to the index of it in symmetrized k-point list
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param nuChi0Eigscomm the trial vector communicator (nuChi0Eigscomm) of this processor
 * @param nuChi0EigsBridgeComm the communicator to connect all processors having the same rank in their nu chi0 eigs communicators
 * @param nuChi0EigscommIndex the index of the nu chi0 eigs communicator
 * @param rank0nuChi0EigscommInWorld global rank of the 0th processor of the nu chi0 eigs communicator
 * @param symk1 the 1st reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 2nd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 3rd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param kPqSymList the list mapping the coord of k-point (from complete k-point list) + q-point (from symmetrized k-point list without shift)
 * @param Nkpts_sym the total number of k-points in symmetrized k-point list to its corresponding symmetrized k-point index.
 */
void restore_electronicGroundState(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld,
     double *symk1, double *symk2, double *symk3, int **kPqSymList, int Nkpts_sym);

/**
 * @brief restore all orbitals from the previous K-S calculation into the parallelization framework of every trial vector communicator (nuChi0Eigscomm)
 *        ask all processors in the 0th trial vector communicator (nuChi0Eigscomm) to read their bands in .orbit files, then broadcast the bands to
 *        other trial vector communicators
 * 
 * @param pSPARC pointer to SPARC_OBJ
 * @param nuChi0Eigscomm the trial vector communicator (nuChi0Eigscomm) of this processor
 * @param nuChi0EigsBridgeComm the communicator to connect all processors having the same rank in their nu chi0 eigs communicators
 * @param nuChi0EigscommIndex the index of the nu chi0 eigs communicator
 * @param rank0nuChi0EigscommInWorld global rank of the 0th processor of the nu chi0 eigs communicator
 * @param symk1 the 1st reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 2nd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 3rd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param kPqSymList the list mapping the coord of k-point (from complete k-point list) + q-point (from symmetrized k-point list without shift)
*/
void restore_orbitals(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld, double *symk1, double *symk2, double *symk3, int **kPqSymList);

/**
 * @brief ask all processors in the 0th trial vector communicator (nuChi0Eigscomm) to read their bands in .orbit files. Every processor has their 
 *        own spin, k-point and band indices. The processor will find its own corresponding .orbit files and read them.
 * @param pSPARC pointer to SPARC_OBJ
 * @param nuChi0Eigscomm the trial vector communicator (nuChi0Eigscomm) of this processor
*/
void read_orbitals_distributed_gamma_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm);

/**
 * @brief read a designated bands from its own .orbit file.
 * @param pSPARC pointer to SPARC_OBJ
 * @param band global index of the band (not local index in the bandcomm!)
 * @param spin global spin of the band
 * @param domainSubarray the band subarray of the processor saved (which part of the band is saved in this processor). In RPA calculation,
 *                       since there is no domain parallelization, it is not a problem
 * @param readXorb the array saving bands
*/
void read_orbitals_distributed_real_RPA(SPARC_OBJ *pSPARC, int band, int spin, MPI_Datatype domainSubarray, double *readXorb);

/**
 * @brief ask all processors in the 0th trial vector communicator (nuChi0Eigscomm) to read their bands in .orbit files. Every processor has their 
 *        own spin, k-point and band indices. The processor will find its own corresponding .orbit files and read them.
 * @param pSPARC pointer to SPARC_OBJ
 * @param nuChi0Eigscomm the trial vector communicator (nuChi0Eigscomm) of this processor
 * @param symk1 the 1st reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 2nd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param symk2 the 3rd reduced coord list of symmetrized k-points (not complete k-points!)
 * @param kPqSymList the list mapping the coord of k-point (from complete k-point list) + q-point (from symmetrized k-point list without shift)
*/
void read_orbitals_distributed_kpt_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *symk1, double *symk2, double *symk3, int **kPqSymList);

/**
 * @brief read a designated bands from its own .orbit file.
 * @param pSPARC pointer to SPARC_OBJ
 * @param flagSymKpt 1 if the .orbit file saves the band; 0 if the .orbit file saves the complex conjugate of the band
 * @param kpt  global index of the k-point
 * @param symk1 the 1st reduced coord of the k-point
 * @param symk2 the 2nd reduced coord of the k-point
 * @param symk2 the 3rd reduced coord of the k-point
 * @param band global index of the band (not local index in the bandcomm!)
 * @param spin global spin of the band
 * @param domainSubarray the band subarray of the processor saved (which part of the band is saved in this processor). In RPA calculation,
 *                       since there is no domain parallelization, it is not a problem
 * @param readXorb_kpt the array saving bands
*/
void read_orbitals_distributed_complex_RPA(SPARC_OBJ *pSPARC, int flagSymKpt, int kpt, double symk1, double symk2, double symk3, int band, int spin, MPI_Datatype domainSubarray, double _Complex *readXorb_kpt);

void restore_electronDensity(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int rank0nuChi0EigscommInWorld);

void restore_eigval_occ(SPARC_OBJ* pSPARC, MPI_Comm nuChi0Eigscomm, MPI_Comm nuChi0EigsBridgeComm, int nuChi0EigscommIndex, int Nkpts_sym, int **kPqSymList);

void read_eigval_occ(char *inputEigsFnames, int Nspin, int Nkpts_sym, int Nstates, double *coordsKptsSym, double *eigsKptsSym, double *occsKptsSym);

void find_eigval_occ_spin_kpts(SPARC_OBJ *pSPARC, int Nkpts_sym, double *coordsKptsSym, double *eigsKptsSym, double *occsKptsSym, int **kPqSymList);

void Transfer_Veff_loc_RPA(SPARC_OBJ *pSPARC, MPI_Comm nuChi0Eigscomm, double *Veff_phi_domain, double *Veff_psi_domain);
#endif