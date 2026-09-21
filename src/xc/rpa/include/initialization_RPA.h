#ifndef INITIALRPA 
#define INITIALRPA

#include "isddft.h"
#include "main_RPA.h"

/**
 * @brief   Initializes parameters and objects of RPA_OBJ
 *
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 * @param pRPA  pointer to RPA_OBJ
 * @param argc  The number of arguments that are passed to the program.
 * @param argv  The array of strings representing command line arguments.
 */
void initialize_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int argc, char* argv[]);

/**
 * @brief   Create MPI struct type RPA_INPUT_MPI.
 *
 *          This function creates a user-defined new MPI datatype, by sending
 *          and receiving it, all processors get input variables for RPA
 *
 * @param RPA_INPUT_MPI (output)    The pointer to the new MPI datatype.
 */
void RPA_Input_MPI_create(MPI_Datatype *RPA_INPUT_MPI);

/**
 * @brief   Set default values for variables in RPA_OBJ
 *
 * @param pRPA_Input  The pointer that points to RPA_INPUT_OBJ type structure.
 * @param Nstates    the amount of states computed in the previous K-S DFT calculation
 * @param Nd         the amount of finite difference gridpoints in the system
 */
void set_RPA_defaults(RPA_INPUT_OBJ *pRPA_Input, int Nstates, int Nd);

/**
 * @brief   Read .rpa input file.
 *
 *          This function reads the input file .rpa and saves variables into the 
 *          structure pRPA_Input. Note that only root process (rank = 0)
 *          will call this function to read data from the input file, 
 *          which will later be broadcasted to all processes.
 *
 * @param pRPA_Input  The pointer that points to RPA_INPUT_OBJ type structure.
 */
void read_RPA_inputs(RPA_INPUT_OBJ *pRPA_Input);

/**
 * @brief   Copy variables read from input file .rpa into struct RPA.
 *
 * @param pRPA  (out) pointer to RPA_OBJ
 * @param pRPA_Input  The pointer that points to RPA_INPUT_OBJ type structure.
 */
void RPA_copy_inputs(RPA_OBJ *pRPA, RPA_INPUT_OBJ *pRPA_Input);

/**
 * @brief   Compute values of omegas to make integration over Im axis
 *
 * @param omega  (out) array saving values of integration points omega
 * @param omega01 (out) Gauss-Legendre integration is applied, they are omega values mapped into 0~1 interval
 * @param omegaWts (out) weights of integration points omega
 * @param Nomega the amount of omegas used in RPA calculation
 */
void set_omegas(double *omega, double *omega01, double *omegaWts, int Nomega);

/**
 * @brief   Write the initialized RPA parameters into the output file.
 *
 * @param pRPA  (out) pointer to RPA_OBJ
 * @param isGammaPoint flag showing the system has only gamma-point
 * @param nspin number of spin handled by the processor (Sternheimer equation distributions)
 * @param nstates number of orbitals handled by the processor
 * @param Nd_d_dmcomm Amount of fd grid points in the dmcomm of this processor, currently it is equal to Nd
 */
void write_settings(RPA_OBJ *pRPA, int isGammaPoint, int nspin, int nstates, int Nd_d_dmcomm);

/**
 * @brief   Compute the needed eigenvalues of the operator sqrt(nu) for all q-points
 * @param pSPARC    The pointer that points to SPARC_OBJ type structure SPARC.
 * @param Nqpts_sym number of q-points in the q-point list
*/
void compute_pois_kron_cons_RPA(SPARC_OBJ *pSPARC, int Nqpts_sym);

#endif