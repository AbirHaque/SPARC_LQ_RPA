#ifndef PARALLEL_RPA
#define PARALLEL_RPA

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief   Setup nuChi0Eigscomm communicators in RPA_OBJ. the communicator saving some deltaV vectors (eigvectors of nu chi0), 
 *          every nuChi0Eigscomm has a complete pSPARC saving all information about the previous K-S DFT calculation
 *          
 * @param pRPA pointer to RPA_OBJ
 * @param Nspin the amount of spin in the previous K-S DFT calculation
 * @param Nkpts the amount of k-points in the previous K-S DFT calculation, remember: it is whole k-point grid, not after time-reverse symmetric reduction
 * @param Nstates the amount of states in the previous K-S DFT calculation
 * @param Nd total amount of finite difference points in the cell
 */
void Setup_Comms_RPA(RPA_OBJ *pRPA, int Nspin, int Nkpts, int Nstates, int Nd);

/**
 * @brief   Calculate a good amount of nuChi0Eigscomms: all eigenpairs of nu chi0 operators, or deltaV vectors, will be divided into npnuChi0Neig parts,
 *          and each part will be handled by a nuChi0Eigscomm
 *          
 * @param nprocWorld size of COMM_WORLD, total number of processors
 * @param rankWorld rank of the current processor in COMM_WORLD
 * @param nuChi0Neig the total number of eigenpairs of nu chi0 operator to be computed
 * @param npnuChi0Neig (out) divisor of all eigenpairs of nu chi0 operator
 */
void dims_divide_Eigs(int nprocWorld, int rankWorld, int nuChi0Neig, int *npnuChi0Neig);

/**
 * @brief   In a special case where input divisor of objects surpasses the amount of objects, or surpass the number of processors, 
 *          the divisor will be modified
 *      
 * @param nObjectInTotal the number of objects to be divided
 * @param sizeFatherComm the amount of processors in FatherComm (at here it is COMM_WORLD), to be distributed into subComms (at here it is nuChi0Eigscomm)
 * @param npInput input divisor of objects. If it is larger than min(sizeFatherComm, nObjectInTotal), it will be replaced
*/
int judge_npObject(int nObjectInTotal, int sizeFatherComm, int npInput);

/**
 * @brief   Distribute objects (at here, they are eigenpairs of nu chi0 operator) into all subcomms (at here, they are nuChi0Eigscomms).
 *
 * @param nObjectInTotal the number of objects (eigenpairs of nu chi0 operators) to be divided
 * @param npObject divisor of objects, or the total number of subcomms (nuChi0Eigscomms)
 * @param rankFatherComm rank of the current processor in FatherComm (at here it is COMM_WORLD)
 * @param sizeFatherComm the amount of processors in FatherComm (at here it is COMM_WORLD), to be distributed into subComms (at here it is nuChi0Eigscomm)
 * @param commIndex (out) the index of subComm (at here it is nuChi0Eigscomm) which this processor is assigned
 * @param objectStartIndex (out) the start index of distributed object (eigenpairs of nu chi0 operators) handled by subComm of this processor
 * @param objectEndIndex (out) the end index of distributed object (eigenpairs of nu chi0 operators) handled by subComm of this processor
 */
int distribute_comm_load(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex);

/**
 * @brief   Distribute objects (at here, they are eigenpairs of nu chi0 operator) into all subcomms (at here, they are nuChi0Eigscomms).
 *
 * @param nObjectInTotal the number of objects (eigenpairs of nu chi0 operators) to be divided
 * @param npObject divisor of objects, or the total number of subcomms (nuChi0Eigscomms)
 * @param rankFatherComm rank of the current processor in FatherComm (at here it is COMM_WORLD)
 * @param sizeFatherComm the amount of processors in FatherComm (at here it is COMM_WORLD), to be distributed into subComms (at here it is nuChi0Eigscomm)
 * @param commIndex (out) the index of subComm (at here it is nuChi0Eigscomm) which this processor is assigned
 * @param objectStartIndex (out) the start index of distributed object (eigenpairs of nu chi0 operators) handled by subComm of this processor
 * @param objectEndIndex (out) the end index of distributed object (eigenpairs of nu chi0 operators) handled by subComm of this processor
 */
int distribute_comm_load_LQ(int nObjectInTotal, int npObject, int rankFatherComm, int sizeFatherComm, int *commIndex, int *objectStartIndex, int *objectEndIndex);


/**
 * @brief   Setup nuChi0BlacsComm communicators in RPA_OBJ. the communicator connect processors in all nuChi0Eigscomms having the same rank.
 *          However, only processors in valid SPARC_dmcomm will be connected, since only they will join the RPA calculation.
 *          Then the contents of processors, and descripters of matrices to be used by blacs functions will be set
 *      
 * @param pRPA pointer to RPA_OBJ
 * @param flagNoDmcomm if this flag is 1, the processor does not have valid domain communicator
 * @param DMnd the length of a band psi saved in the current processor, psi is distributed in psi domain
 * @param Nspinor_spincomm the coefficient of spin and spin-orbit coupling on vector length. spin/spin-orbit coupling: 2; otherwise 1
 * @param Nspinor_eig the coefficient of spin and spin-orbit coupling. UNNECESSARY? Number of y not related to spin
 * @param Nd total amount of finite difference points in the cell
 * @param MAX_NS the limit of eigenproblem size using serial eigensolver. If the eigenproblem size is larger than that, parallel eigensolver will be used
 * @param eig_paral_blksz the block size used for distributing the eigenproblem
 * @param isGammaPoint flag showing the system has only gamma-point
 */    
void setup_blacsComm_RPA(RPA_OBJ *pRPA, int flagNoDmcomm, int DMnd, int Nspinor_spincomm, int Nspinor_eig, 
    int Nd, int MAX_NS, int eig_paral_blksz, int isGammaPoint, int printFlag);

/**
 * @brief generate permute matrix to redistribute trial deltaV vectors in a cyclix sequence in every nuChi0Eigscomm
 *        to weaken the influence of load-imbalance problem in deltaV vectors
 *
 * @param desc_permute descripter of permutation matrix distributed in 1D blacs context in blacscomm (nuChi0EigsBridgeComm), column-wise partition
 * @param desc_permute_BLCYC descripter of permutation matrix in row-wise cyclix partition
 * @param rPermute local row of permutation matrix in this processor, it is equal to the number of deltaV vectors, nuChi0Neig
 * @param cPermute local col of permutation matrix in this processor
 * @param permute local permutation matrix
 * @param permute_BLCYC local permutation matrix row-wise cyclix partitioned in blacscomm
 * @param ictxt_blacs 1D context in blacs communicator
 * @param nuChi0BlacsComm communicator making operations related to projected operator Hp = VT(nu^0.5 chi0 nu^0.5)V (1st nuChi0EigsBridgeComm)
 */   
void generate_permute_matrix(int *desc_permute, int *desc_permute_BLCYC, int rPermute, int cPermute, 
    double *permute, double *permute_BLCYC, int ictxt_blacs, MPI_Comm nuChi0BlacsComm);

/**
 * @brief generate permute matrix to redistribute trial deltaV vectors in a cyclix sequence in every nuChi0Eigscomm
 *        to weaken the influence of load-imbalance problem in deltaV vectors
 *
 * @param desc_permute descripter of permutation matrix distributed in 1D blacs context in blacscomm (nuChi0EigsBridgeComm), column-wise partition
 * @param desc_permute_BLCYC descripter of permutation matrix in row-wise cyclix partition
 * @param rPermute local row of permutation matrix in this processor, it is equal to the number of deltaV vectors, nuChi0Neig
 * @param cPermute local col of permutation matrix in this processor
 * @param permute_kpt local permutation matrix, column-wise partition
 * @param permute_BLCYC_kpt local permutation matrix row-wise cyclix partitioned in blacscomm
 * @param ictxt_blacs 1D context in blacs communicator
 * @param nuChi0BlacsComm communicator making operations related to projected operator Hp = VT(nu^0.5 chi0 nu^0.5)V (1st nuChi0EigsBridgeComm)
 */   
void generate_permute_matrix_kpt(int *desc_permute, int *desc_permute_BLCYC, int rPermute, int cPermute, 
    double _Complex *permute_kpt, double _Complex *permute_BLCYC_kpt, int ictxt_blacs, MPI_Comm nuChi0BlacsComm);

/**
 * @brief Setup HpDiagComm communicators in RPA_OBJ. The set of communicators solve diagonal entries in ln(1-Hp) + Hp in parallel.
 *        Only used in SQ method.
 * @param pRPA pointer to RPA_OBJ
 * @param eig_paral_blksz the block size used for distributing projected operator Hp in HpDiagComm
 * @param printFlag flag to print mid variables
 */
void setup_HpDiagComm_RPASQ(RPA_OBJ *pRPA, int eig_paral_blksz, int isGammaPoint, int printFlag);

void generate_permute_matrix_SQ(int *desc_permute_BLCYC, int sizePermute, int rPermute_BLCYC, int cPermute_BLCYC, 
    double *permute_BLCYC, int *desc_pTp, double *pTp, int ictxt_blacs, MPI_Comm nuChi0BlacsComm);
#endif