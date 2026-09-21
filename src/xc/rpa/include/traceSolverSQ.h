#ifndef TRACESOLVER
#define TRACESOLVER

/**
 * @brief transfer the projected operator from blacscomm (1st nuChi0EigsBridgeComm) to HpDiagComm
 *
 * @param nuChi0Neig amount of trial vectors V, size of Hp = V'XV
 * @param Hp projected operator Hp = V'XV = V'(nu^0.5 chi0 nu^0.5)V distributed block cyclically in nuChi0BlacsComm (1st nuChi0EigsBridgeComm)
 * @param input_desc_Hp_BLCYC descripter of Hp operator in nuChi0BlacsComm
 * @param HpSQ projected operator Hp = V'XV distributed in 1D topology in HpDiagComm
 * @param input_desc_HpSQ descripter of Hp operator in HpDiagComm
 * @param nrHpSQ number of rows of local part of Hp in HpDiagComm
 * @param ncHpSQ number of cols of local part of Hp in HpDiagComm
 * @param nuChi0EigsBridgeCommIndex index of nuChi0EigsBridgeComm, only processors in 1st nuChi0EigsBridgeComm take part in the transfer
 * @param HpDiagCommIndex index of HpDiagComm, only 1st HpDiagComm receive Hp, then 1st HpDiagComm broadcast Hp to other HpDiagComms
 * @param ctxtWorld handle of context of COMM_WORLD, for pdgemr2d_
 * @param HpDiagBridgeComm the communicator to connect all processors having the same rank in their HpDiagComm
 * @param rankHpDiagComm rank of the processor in HpDiagComm
 * @param printFlag flag to print mid variables
 */
void transfer_Hp_HpDiagComms(int nuChi0Neig, double *Hp, int *input_desc_Hp_BLCYC, double *HpSQ, int *input_desc_HpSQ, int nrHpSQ, int ncHpSQ,
    int nuChi0EigsBridgeCommIndex, int HpDiagCommIndex, int ctxtWorld, MPI_Comm HpDiagBridgeComm, int rankHpDiagComm, int printFlag);

/**
 * @brief Make all Lanczos decompositions starting from all assigned unit vector Vk = [e_x, ..., e_y] on Hp together
 *        to get all tridiagonal matrices T
 *
 * @param nuChi0Neig amount of trial vectors V, size of Hp = V'XV
 * @param HpSQ projected operator Hp = V'XV distributed in 1D topology in HpDiagComm
 * @param desc_HpSQ descripter of Hp operator in HpDiagComm
 * @param HpDiagStartIndex the start index of the diagonal entries to be solved by this HpDiagComm
 * @param HpDiagEndIndex the end index of the diagonal entries to be solved by this HpDiagComm
 * @param vkStart the starting row index of Vk assigned to this processor
 * @param vkEnd the ending row index of Vk assigned to this processor
 * @param nrvk number of rows of Vk assigned to this processor
 * @param desc_vk descripter of Vk
 * @param nrvk_BLCYC number of rows of Vk in cyclix distribution assigned to this processor
 * @param desc_vk_BLCYC descripter of Vk_BLCYC, Vk in cyclix distribution 
 * @param pictxt_HpSQ pointer to the context in HpDiagComm
 * @param nplLanczos number of polynomial order in Lanczos decomposition
 * @param TdiagArray array saving diagonals of Ts obtained from all Lanczos decompositions together
 * @param ToffDiagArray array saving off diagonals of Ts obtained from all Lanczos decompositions together
 * @param HpDiagComm HpDiag communicator to solve designated diagonal entries in ln(1 - Hp) + Hp
 * @param HpDiagCommIndex the index of HpDiag communicator
 * @param printFlag flag to print mid variables
 */
int Lanczos_decompose_Hp_group(int nuChi0Neig, double *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex, 
    int vkStart, int vkEnd, int nrvk, int *desc_vk, int nrvk_BLCYC, int *desc_vk_BLCYC, int *pictxt_HpSQ,
    int nplLanczos, double *TdiagArray, double *ToffDiagArray, MPI_Comm HpDiagComm, int HpDiagCommIndex, int printFlag);

/**
 * @brief diagonalize tridiagonal matrix T from Lanczos decomposition and get the diagonal entry in ln(1-Hp) + Hp
 *
 * @param finalNpl the exact order of Lanczos decomposition. In most cases it is equal to nplLanczos
 * @param diagIndex the index of the diagonal entry to be solved
 * @param TdiagArray array saving diagonals of Ts obtained from all Lanczos decompositions together
 * @param ToffDiagArray array saving off diagonals of Ts obtained from all Lanczos decompositions together
 * @param TeigVals eigenvalues of T
 * @param TeigVecs eigenvectors of T
 * @param rankHpDiagComm rank of the processor in its HpDiagComm
 * @param HpDiagCommIndex the index of HpDiag communicator
 * @param printFlag flag to print mid variables
 */
double diagonalize_T_get_diag_fHp(int finalNpl, int diagIndex, double *TdiagArray, double *ToffDiagArray, double *TeigVals, double *TeigVecs, 
    int rankHpDiagComm, int HpDiagCommIndex, int printFlag);

    /**
 * @brief transfer the projected operator from blacscomm (1st nuChi0EigsBridgeComm) to HpDiagComm
 *
 * @param nuChi0Neig amount of trial vectors V, size of Hp = V'XV
 * @param Hp projected operator Hp = V'XV = V'(nu^0.5 chi0 nu^0.5)V distributed block cyclically in nuChi0BlacsComm (1st nuChi0EigsBridgeComm)
 * @param input_desc_Hp_BLCYC descripter of Hp operator in nuChi0BlacsComm
 * @param HpSQ projected operator Hp = V'XV distributed in 1D topology in HpDiagComm
 * @param input_desc_HpSQ descripter of Hp operator in HpDiagComm
 * @param nrHpSQ number of rows of local part of Hp in HpDiagComm
 * @param ncHpSQ number of cols of local part of Hp in HpDiagComm
 * @param nuChi0EigsBridgeCommIndex index of nuChi0EigsBridgeComm, only processors in 1st nuChi0EigsBridgeComm take part in the transfer
 * @param HpDiagCommIndex index of HpDiagComm, only 1st HpDiagComm receive Hp, then 1st HpDiagComm broadcast Hp to other HpDiagComms
 * @param ctxtWorld handle of context of COMM_WORLD, for pdgemr2d_
 * @param HpDiagBridgeComm the communicator to connect all processors having the same rank in their HpDiagComm
 * @param rankHpDiagComm rank of the processor in HpDiagComm
 * @param printFlag flag to print mid variables
 */
void transfer_Hp_HpDiagComms_kpt(int nuChi0Neig, double _Complex *Hp, int *input_desc_Hp_BLCYC, double _Complex *HpSQ, int *input_desc_HpSQ, int nrHpSQ, int ncHpSQ,
    int nuChi0EigsBridgeCommIndex, int HpDiagCommIndex, int ctxtWorld, MPI_Comm HpDiagBridgeComm, int rankHpDiagComm, int printFlag);

/**
 * @brief Make all Lanczos decompositions starting from all assigned unit vector Vk = [e_x, ..., e_y] on Hp together
 *        to get all tridiagonal matrices T
 *
 * @param nuChi0Neig amount of trial vectors V, size of Hp = V'XV
 * @param HpSQ projected operator Hp = V'XV distributed in 1D topology in HpDiagComm
 * @param desc_HpSQ descripter of Hp operator in HpDiagComm
 * @param HpDiagStartIndex the start index of the diagonal entries to be solved by this HpDiagComm
 * @param HpDiagEndIndex the end index of the diagonal entries to be solved by this HpDiagComm
 * @param vkStart the starting row index of Vk assigned to this processor
 * @param vkEnd the ending row index of Vk assigned to this processor
 * @param nrvk number of rows of Vk assigned to this processor
 * @param desc_vk descripter of Vk
 * @param nrvk_BLCYC number of rows of Vk in cyclix distribution assigned to this processor
 * @param desc_vk_BLCYC descripter of Vk_BLCYC, Vk in cyclix distribution 
 * @param pictxt_HpSQ pointer to the context in HpDiagComm
 * @param nplLanczos number of polynomial order in Lanczos decomposition
 * @param TdiagArray array saving diagonals of Ts obtained from all Lanczos decompositions together
 * @param ToffDiagArray array saving off diagonals of Ts obtained from all Lanczos decompositions together
 * @param HpDiagComm HpDiag communicator to solve designated diagonal entries in ln(1 - Hp) + Hp
 * @param HpDiagCommIndex the index of HpDiag communicator
 * @param printFlag flag to print mid variables
 */
int Lanczos_decompose_Hp_group_kpt(int nuChi0Neig, double _Complex *HpSQ, int *desc_HpSQ, int HpDiagStartIndex, int HpDiagEndIndex, 
    int vkStart, int vkEnd, int nrvk, int *desc_vk, int nrvk_BLCYC, int *desc_vk_BLCYC, int *pictxt_HpSQ,
    int nplLanczos, double _Complex *TdiagArrayGroup, double _Complex *ToffDiagArrayGroup, MPI_Comm HpDiagComm, int HpDiagCommIndex, int printFlag);

/**
 * @brief diagonalize tridiagonal matrix T from Lanczos decomposition and get the diagonal entry in ln(1-Hp) + Hp
 *
 * @param finalNpl the exact order of Lanczos decomposition. In most cases it is equal to nplLanczos
 * @param diagIndex the index of the diagonal entry to be solved
 * @param TdiagArray array saving diagonals of Ts obtained from all Lanczos decompositions together
 * @param ToffDiagArray array saving off diagonals of Ts obtained from all Lanczos decompositions together
 * @param TeigVals eigenvalues of T
 * @param TeigVecs eigenvectors of T
 * @param rankHpDiagComm rank of the processor in its HpDiagComm
 * @param HpDiagCommIndex the index of HpDiag communicator
 * @param printFlag flag to print mid variables
 */
double diagonalize_T_get_diag_fHp_kpt(int finalNpl, int diagIndex, double _Complex *TdiagArray, double _Complex *ToffDiagArray, double *TeigVals, double _Complex *TeigVecs, 
    int rankHpDiagComm, int HpDiagCommIndex, int printFlag);

#endif