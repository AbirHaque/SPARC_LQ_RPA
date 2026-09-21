#ifndef KPTGPT
#define KPTGPT

#include "main_RPA.h"

/**
 * @brief Transfer the k-point grid after symmetrical reduction from pSPARC to pRPA. SPARC code is based on k-point grid after 
 *        symmetrical reduction, but in RPA calculation all k-points should be calculated for a designated q-point. So the 
 *        complete k-point grid should replace the k-point grid after symmetrical reduction in pSPARC. However, the symmetrical
 *        reduced k-point grid is necessary for reading the orbital files and eigenvalues. They are saved in pRPA instead.
 * @param pSPARC pointer to SPARC_OBJ
 * @param pRPA pointer to RPA_OBJ
 */
void transfer_kpoints(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA);

/**
 * @brief Recalculate the complete k-point grid, and replace the symmetrical reduced k-point grid in pSPARC.
 * @param pSPARC pointer to SPARC_OBJ
 */
void recalculate_kpoints(SPARC_OBJ *pSPARC);

/**
 * @brief Set q-point grids. Q-point grids are symmetric reduced k-points without shift.
 * @param qptWts (output) the weight of q-points. The value is either 1 or 2. To use it, dividing it by the total number of k-point
 *               in complete k-point grid is necessary.
 * @param q1 (output) the coord of q-points in 1st reciprocal latvec direction. Don't miss it with reduced coordinate in M-SPARC.
 * @param q2 (output) the coord of q-points in 2nd reciprocal latvec direction. Don't miss it with reduced coordinate in M-SPARC.
 * @param q3 (output) the coord of q-points in 3rd reciprocal latvec direction. Don't miss it with reduced coordinate in M-SPARC.
 * @param Kx the number of k-points in 1st direction
 * @param Ky the number of k-points in 2nd direction
 * @param Kz the number of k-points in 3rd direction
 * @param Lx the length of 1st letvec in real space
 * @param Ly the length of 2nd letvec in real space
 * @param Lz the length of 3rd letvec in real space
 */
int set_qpoints(double *qptWts, double *q1, double *q2, double *q3, int Kx, int Ky, int Kz, double Lx, double Ly, double Lz);

/**
 * @brief Generate three tables. Given a designated q-point, all k-points find the index of their k+q pairs and k-q pairs by the tables.
 * @param Nkpts_sym number of k-points in the symmetric reduced k-point list
 * @param k1sym the coord of symmetric reduced k-points in 1st reciprocal latvec direction.
 * @param k2sym the coord of symmetric reduced k-points in 2nd reciprocal latvec direction.
 * @param k3sym the coord of symmetric reduced k-points in 3rd reciprocal latvec direction.
 * @param Nkpts number of k-points in the complete k-point list
 * @param k1 the coord of complete k-points in 1st reciprocal latvec direction.
 * @param k2 the coord of complete k-points in 2nd reciprocal latvec direction.
 * @param k3 the coord of complete k-points in 3rd reciprocal latvec direction.
 * @param Nqpts_sym number of q-points in the q-point list
 * @param q1 the coord of q-points in 1st reciprocal latvec direction.
 * @param q2 the coord of q-points in 2nd reciprocal latvec direction.
 * @param q3 the coord of q-points in 3rd reciprocal latvec direction.
 * @param Lx the length of 1st letvec in real space
 * @param Ly the length of 2nd letvec in real space
 * @param Lz the length of 3rd letvec in real space
 * @param kPqSymList (output) the list of k+q in symmetric reduced k-point list. Index of k+q is kPqSymList[kIndex][qIndex+1].
 *                   Original: positive; Reflection: negative 
 * @param kPqList (output) the list of k+q in complete k-point list. Index of k+q is kPqList[kIndex][qIndex]
 * @param kMqList (output) the list of k-q in complete k-point list. Index of k-q is kMqList[kIndex][qIndex]
 */
void set_kPq_kMq_lists(int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym, int Nkpts, double *k1, double *k2, double *k3, 
    int Nqpts_sym, double *q1, double *q2, double *q3, double Lx, double Ly, double Lz, int **kPqSymList, int **kPqList, int **kMqList);

/**
 * @brief given the coord of a k-point in reciprocal latvec directions, its index in the symmetric reduced k-point list is found. If it is 
 *        saved in the list, the index return in positive; if its reflection is saved in the list, the index return in negative
 * @param k1k2k3 the coord of input k-point
 * @param Nkpts_sym number of k-points in the symmetric reduced k-point list
 * @param k1sym the coord of symmetric reduced k-points in 1st reciprocal latvec direction.
 * @param k2sym the coord of symmetric reduced k-points in 2nd reciprocal latvec direction.
 * @param k3sym the coord of symmetric reduced k-points in 3rd reciprocal latvec direction.
 * @param Lx the length of 1st letvec in real space
 * @param Ly the length of 2nd letvec in real space
 * @param Lz the length of 3rd letvec in real space
*/
int find_kpt_sym_index(double k1, double k2, double k3, int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym, double Lx, double Ly, double Lz);

/**
 * @brief given the coord of a k-point in reciprocal latvec directions, find whether or not the k-point is saved in the symmetric reduced k-point list.
 *        If so, return its index.
 * @param k1Coord the coord of input k-point in 1st reciprocal latvec direction.
 * @param k2Coord the coord of input k-point in 2nd reciprocal latvec direction.
 * @param k3Coord the coord of input k-point in 3rd reciprocal latvec direction.
 * @param Nkpts_sym number of k-points in the symmetric reduced k-point list
 * @param k1sym the coord of symmetric reduced k-points in 1st reciprocal latvec direction.
 * @param k2sym the coord of symmetric reduced k-points in 2nd reciprocal latvec direction.
 * @param k3sym the coord of symmetric reduced k-points in 3rd reciprocal latvec direction.
*/
int find_kpt_sym_1to1(double k1Coord, double k2Coord, double k3Coord, int Nkpts_sym, double *k1sym, double *k2sym, double *k3sym);
#endif