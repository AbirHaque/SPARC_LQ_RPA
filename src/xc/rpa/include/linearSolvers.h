#ifndef LINEARSOLVERRPA
#define LINEARSOLVERRPA

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief  Conjugate Orthogonal Conjugate Gradient (COCG) method for solving a set of Sternheimer equations with the same LHS.
 *         This linear solver does not accept vector distribution over many processors for now.
 * 
 * @param lhsfun (input) the left-hand side of Sternheimer equation, a function mapping a vector to a vector
 * @param pSPARC pointer to SPARC_OBJ (just for printing mid variables)
 * @param spn_i spin index of the band to be computed in this set of Sternheimer equation
 * @param psi the band vector of this set of Sternheimer equations, only one band!
 * @param epsilon the eigenvalue of this band psi
 * @param omega the omega in this set of Sternheimer equations
 * @param flagPQ the flag to apply linear operator Q(|psi><psi|) and P on both sides of Sternheimer equations
 * @param deltaPsisReal (output) the real part of solution vectors
 * @param deltaPsiImag (output) the imaginary part of solution vectors
 * @param SternheimerRhs (input) the RHS vectors of Sternheimer equations
 * @param rhsBlockSize the amount of RHS vectors in this Sternheimer equation set
 * @param flagPreconditioner the flag to decide whether or not applying Laplace preconditioner
 * @param precond the preconditioner function
 * @param poisC_invE_precond the eigenvalues of the operator (Laplacian - epsilon)^{-1}, used for preconditioner
 * @param Icoeff the coefficient c in preconditioer ((Laplacian - epsilon)^{-1} + c*I), for adding robustness
 * @param RHS_frobnormsq Frobenius norm of RHS
 * @param sternRelativeResTol the relative residual norm tolerance to judge convergence
 * @param maxIter the limit of iteration times
 * @param resNormRecords the record of residuals of all equations in all iterations
 * @param lhsTime the pointer recording the time spent on multiplying LHS by vectors
 * @param solveMuTime the pointer recording the time spent on solving the small linear equations
 * @param multipleTime the pointer recording the time spent on matrix multiplications
 */
int block_COCG(void (*lhsfun)(SPARC_OBJ*, int, double *, double, double, int, double *, double *, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, double *psi, double epsilon, double omega, int flagPQ, double *deltaPsisReal, double *deltaPsisImag,
     double _Complex *SternheimerRhs, int rhsBlockSize,
     int flagPreconditioner, void (*precond)(SPARC_OBJ *, double *, double *, int, double *, double), double *poisC_invE_precond, double Icoeff,
     double RHS_frobnormsq, double sternRelativeResTol, int maxIter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime);


/**
 * @brief  Judge whether the solution of COCG in this iteration has converged by comparing the relative residual of every equation
 *         with the relative residual tolerance.
 * 
 * @param ix the time of iteration
 * @param numVecs the amount of RHS vectors, or "nuChi0EigsAmounts" in function block_COCG
 * @param RHS2norm 2-norm of all RHS vectors
 * @param sternRelativeResTol the relative residual norm tolerance to judge convergence
 * @param resNormRecords the record of residuals of all equations in all iterations
 */
int judge_converge(int ix, int numVecs, const double RHS_frobnormsq, double sternRelativeResTol, const double *resNormRecords);

/**
 * @brief  Coupled Residual Minimization (CRM) method for solving a set of coupled Sternheimer equations with the same LHS.
 *         This solver is for calculating systems with k-points.
 *         This linear solver does not accept vector distribution over many processors for now.
 * 
 * @param lhsfun (input) the left-hand side of Sternheimer equation, a function mapping a vector to a vector
 * @param pSPARC pointer to SPARC_OBJ (just for printing mid variables)
 * @param spn_i spin index of the band to be computed in this set of Sternheimer equation
 * @param kPq the k-point of Hamiltonian in the LHS, which is kpt of the band adding qpt of the chi0 operator
 * @param psi_kpt the band vector of this set of Sternheimer equations, only one band!
 * @param epsilon the eigenvalue of this band psi
 * @param omega the omega in this set of Sternheimer equations
 * @param flagPQ the flag to apply linear operator Q(|psi><psi|) and P on both sides of Sternheimer equations
 * @param deltaPsisPlus (output) the solutions of the Sternheimer equation with +i\omega on LHS
 * @param deltaPsisMinus (output) the solutions of the Sternheimer equation with -i\omega on LHS
 * @param SternheimerRhs (input) the RHS vectors of Sternheimer equations
 * @param rhsBlockSize the amount of RHS vectors in this Sternheimer equation set
 * @param RHS_frobnormsq Frobenius norm of RHS
 * @param sternRelativeResTol the relative residual norm tolerance to judge convergence
 * @param maxInnerIter the limit of inner iteration time; once limit is reached, the CRM will restart with initial guess <- current results
 * @param maxOuterIter the limit of outer iteration time
 * @param resNormRecords the record of residuals of all equations in all iterations
 * @param lhsTime the pointer recording the time spent on multiplying LHS by vectors
 * @param solveMuTime the pointer recording the time spent on solving the small linear equations
 * @param multipleTime the pointer recording the time spent on matrix multiplications
 */
int block_CRM(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
     SPARC_OBJ* pSPARC, int spn_i, int kPq, double _Complex *psi_kpt, double epsilon, double omega, int flagPQ, double _Complex *deltaPsisPlus, double _Complex *deltaPsisMinus,
     double _Complex *SternheimerRhs, int rhsBlockSize,
     // int flagPreconditioner, void (*precond)(SPARC_OBJ *, double *, double *, int, double *, double), double *poisC_invE_precond, double Icoeff,
     double RHS_frobnormsq, double sternRelativeResTol, int maxInnerIter, int maxOuterIter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime);

int AAR_sternheimer_kpt(void (*lhsfun)(SPARC_OBJ*, int, int, double, double, double _Complex*, double _Complex*, int),
    SPARC_OBJ* pSPARC, int spn_i, int kPq, double _Complex *psi_kpt, double epsilon, double omega, int flagPQ, double _Complex *x,
    double _Complex *SternheimerRhs,
    // void (*precond_fun)(SPARC_OBJ *,int,double,double *,double*,MPI_Comm), double c, 
    double RHS_frobnormsq, double tol, int max_iter, double *resNormRecords, double *lhsTime, double *solveMuTime, double *multipleTime,
    double omegaR, double betaA, int m, int p);

void  AndersonExtrapolation_complex(
        const int N, const int m, double _Complex *x_kp1, const double _Complex *x_k, 
        const double _Complex *f_k, const double _Complex *X, const double _Complex *F, 
        const double _Complex betaA, MPI_Comm comm);

#endif