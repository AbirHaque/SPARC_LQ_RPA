#ifndef STERNEQ
#define STERNEQ

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief Solve all Sternheimer eq.s of trial vector dVs (from its nuChi0Eigscomm) and bands psis (from its bandcomm) assigned to this processor
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param omegaIndex the index of omega of the current nu chi0 operator
 * @param nuChi0EigsAmount amount of solved eigenvalues in this nuChi0Eigscomm, at here it is the amount of DV vectors
 * @param DVs (input) deltaVs in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param printFlag flag to print mid variables
 */
void sternheimer_eq_gamma(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int omegaIndex, int nuChi0EigsAmount, double *DVs, int printFlag);

/**
 * @brief Solve a Sternheimer eq. for deltaVs and ONE psi
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param spn_i spin of the input orbital psi_n
 * @param epsilon (input) epsilon_n in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param omega (input) omega in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param flagPQ the flag to use linear operator P (|psi_n><psi_n| / dV) and Q (|psi_n><psi_n|)
 * 
 * @param flagCOCGinitial the flag to compose the initial guess of Sternheimer eq.s by eigenpairs got from K-S calculation
 * @param allXorb_thes all orbitals of the spin from K-S calculation, for generating initial guess
 * @param allLambdas_thes all eigenvalues of orbitals of the spin from K-S calculation, for generating initial guess
 * @param NusedOrbitalForInitGuess number of K-S orbitals used for generating initial guess. Yes, not all of them will be used
 * 
 * @param flagPreconditioner the flag to decide whether or not applying Laplace preconditioner
 * @param poisC_invE_precond the eigenvalues of the operator (Laplacian - epsilon)^{-1}, used for preconditioner
 * @param Icoeff the coefficient c in preconditioer ((Laplacian - epsilon)^{-1} + c*I), for adding robustness
 * 
 * @param deltaPsisReal (output) the real part of solutions of this set of Sternheimer eq.s
 * @param deltaPsisImag (output) the imaginary part of solutions of this set of Sternheimer eq.s
 * @param deltaVs (input) deltaVs in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param psi (input) psi_n in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param bandWeight occupation*spin factor of orbital psi_n
 * @param deltaRhos (output) delta rho vectors from this orbital psi_n perturbed by deltaVs
 * @param nuChi0EigsAmounts the amounts of deltaVs
 * @param sternRelativeResTol the tolerance of MAXIMUM relative residuals. It can be replaced by Ferbenious norm to decrease iteration time
 * @param sternBlockSize the block size of RHS. All RHS [deltaV1 psi_n, deltaV2 psi_n, ..., deltaVn psi_n] are not solved together. They are solved block by block
 * @param printFlag flag to print mid variables
 * @param outputName the file name to record the running time of COCG for each nuChi0Eigscomm
 * @param timeRecorder the array recording the running time each part of COCG spent
 */
void sternheimer_solver_gamma(SPARC_OBJ *pSPARC, int spn_i, double epsilon, double omega, int flagPQ, 
    int flagCOCGinitial, double *allXorb_thes, double *allLambdas_thes, int NusedOrbitalForInitGuess, 
    int flagPreconditioner, double *poisC_invE_precond, double Icoeff,
    double *deltaPsisReal, double *deltaPsisImag, double *deltaVs, double *psi, double bandWeight, double *deltaRhos, int nuChi0EigsAmounts,
    double sternRelativeResTol, int SternBlockSize, int printFlag, char *outputName, double *timeRecorder);

/**
 * @brief obtain the product of the LHS of the Sternheimer equation and any set of complex vectors X, 
 *        LHS at here is (K-S Hamiltonian - epsilon_n + Q, if (flagPQ), |psi_n><psi_n| - i omega)X
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param spn_i spin of the input orbital psi_n
 * @param epsilon (input) epsilon_n in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param omega (input) omega in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param flagPQ the flag to use linear operator P (|psi_n><psi_n| / dV) and Q (|psi_n><psi_n|)
 * @param Xreal (input) real part of the vectors to be left multiplied by LHS
 * @param Ximag (input) imaginary part of the vectors to be left multiplied by LHS
 * @param lhsX (output) the product of the LHS of the Sternheimer equation and complex vectors X
 * @param nuChi0EigsAmounts the amounts of vectors in X
 */
void Sternheimer_lhs(SPARC_OBJ *pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *Xreal, double *Ximag, double _Complex *lhsX, int nuChi0EigsAmounts);

/**
 * @brief compute the initial guess solutions of the Sternheimer equations based on the eigenpairs of Hamiltonian
 *
 * @param pSPARC pointer to SPARC_OBJ
 * @param epsilon (input) epsilon_n in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param omega (input) omega in Sternheimer eq. (Hscf - epsilon_n + i omega) = - deltaV*psi_n
 * @param flagPQ the flag to use linear operator P (|psi_n><psi_n| / dV) and Q (|psi_n><psi_n|)
 * @param flagCOCGinitial the flag to use the initial guess solutions of Sternheimer equations. If it is closed, the initial guesses are set as zero vectors
 * @param NusedOrbital (input) the number of first several orbitals to be used to generate the initial guess solutions. The more orbitals used, the better the initial guesses.
 * @param allXorb_thes all of the solved orbitals of the spin from the Kohn-Sham DFT calculation
 * @param allLambdas_thes all of the solved eigenvalues of the spin from the Kohn-Sham DFT calculation
 * @param SternheimerRhs (input) all RHS of the Sternheimer equation, -psi_n .* [dV1 dV2 ... dVn]
 * @param deltaPsisReal (output) real part of the initial guesses
 * @param deltaPsisImag (output) imaginary part of the initial guesses
 * @param nuChi0EigsAmounts the amounts of vectors in RHS
 */
void set_initial_guess_deltaPsis(SPARC_OBJ *pSPARC, double epsilon, double omega, int flagPQ, int flagCOCGinitial, int NusedOrbital, double *allXorb_thes, double *allLambdas_thes,
    double _Complex *SternheimerRhs, double *deltaPsisReal, double *deltaPsisImag, int nuChi0EigsAmounts);

/**
 * @brief compute the eigenvalues of the coefficient (Laplacian - epsilon)^-1, used for preconditioner
 * 
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 * @param epsilon the eigenvalue of the orbital in Sternheimer equation
 */
void compute_pois_kron_cons_LapPrecond(RPA_OBJ *pRPA, SPARC_OBJ *pSPARC, double epsilon);

/**
 * @brief preconditioner of LHS of Sternheimer equaition. ((Laplacian - epsilon)^{-1} + c*I)
 * 
 * @param pSPARC pointer to SPARC_OBJ
 * @param inputVecs the vectors to be left multiplied by preconditioner, they are residual vectors in COCG
 * @param outputVecs the product of preconditioner and inputVecs
 * @param vecAmount the amount of vectors to be multiplied
 * @param poisC_invE_precond the eigenvalues of the operator (Laplacian - epsilon)^{-1}, used for preconditioner
 * @param Icoeff the coefficient c in preconditioer ((Laplacian - epsilon)^{-1} + c*I), for adding robustness
 */
void laplace_preconditioner(SPARC_OBJ *pSPARC, double *inputVecs, double *outputVecs, int vecsAmount, double *poisC_invE_precond, double Icoeff);

#endif