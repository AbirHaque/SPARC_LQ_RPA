#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#ifdef USE_MKL
#define MKL_Complex16 double _Complex
#include "mkl.h"
#include "mkl_lapacke.h"
#include "blacs.h"     // Cblacs_*
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#endif

#ifdef USE_SCALAPACK
#include <cblas.h>
#include <lapacke.h>
#include "blacs.h"     // Cblacs_*
#include "scalapack.h" // ScaLAPACK functions
#endif

#include "hamiltonianVecRoutines.h"
#include "tools.h"

#include "restoreElectronicGroundState.h"
#include "lanczosQuadrature.h"
#include "collectOrbitals.h"
#include "nuChi0VecRoutines.h"
#include "eigenSolverGamma_RPA.h"
#include "tools_RPA.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef double dft_complex[2];

/**
 * @brief   Lanczos quadrature
 */
double  lanczos_quadrature(const SPARC_OBJ *pSPARC, RPA_OBJ *pRPA,
                            double* a, double* b,
                            int MAXIT, int unit_j, MPI_Comm comm,int omegaIndex) 
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double *qT,*v;
    int i,j,k,DMnd,DMndspe;


    DMndspe=pSPARC->Nd;
    
    qT=(double*)calloc(DMndspe*MAXIT,sizeof(double));
    v=(double*)calloc(DMndspe,sizeof(double));


    qT[unit_j]=1.0;
   
    j=0;
    while (j<MAXIT) 
    {
        cblas_dscal(DMndspe, 0.0, v, 1);
        nuChi0_mult_vectors_gamma(pSPARC, pRPA, omegaIndex, &qT[j*DMndspe], v, 1, 0, 0, pRPA->flagWRITE_NUCHI0_MULT_VECTORS_INFO);
        a[j] = cblas_ddot(DMndspe, v, 1, &qT[j*DMndspe], 1);
        cblas_daxpy(DMndspe, -a[j], &qT[j*DMndspe], 1, v, 1);
        if(j>0){
            cblas_daxpy(DMndspe, -b[j-1], &qT[(j-1)*DMndspe], 1, v, 1);
        }
        if(j<MAXIT-1){
            b[j] = cblas_dnrm2(DMndspe, v, 1);
            cblas_dcopy(DMndspe, v, 1, &qT[(j+1)*DMndspe], 1);
            cblas_dscal(DMndspe, 1.0 / b[j], &qT[(j+1)*DMndspe], 1);
        }
        j++;
    }
    double *T=(double*)calloc(MAXIT*MAXIT, sizeof(double));
    LAPACKE_dstev(LAPACK_COL_MAJOR,'V',MAXIT,a,b,T,MAXIT);   

    double result=0.0;
    for (k=0;k<MAXIT;k++){
        if(a[k]<1.0){
            result+=(log(1.0-a[k])+a[k])*T[0+k*MAXIT]*T[0+k*MAXIT];
        }
    }
    free(qT);
    free(T);
    free(v);
    return result;

}


void from_unit(int unit_j,int nx, int ny, int nz, int* i, int* j, int* k){
    *i=unit_j%nx;
    *j=(unit_j/nx)%ny;
    *k=(unit_j/nx)/ny;
}

double lanczos_quadrature_wrapper(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int n, int m, int omegaIndex){
    double result=0.0;
    int unit_j;
    double *Tdiag=(double*)malloc(m*sizeof(double));
    double *Toffdiag=(double*)malloc((m-1)*sizeof(double));


    int rank,size;
    int start,end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int chunk_size=pSPARC->Nd/size;
    int remainder=pSPARC->Nd%size;
    if(rank<remainder){
        start=rank*(chunk_size+1);
        end=start+(chunk_size+1);
    }
    else{
        start=rank*chunk_size+remainder;
        end=start+chunk_size;
    }   

    if(pRPA->flagWRITE_DIAGS){
        int local_count = end - start;
        int vals_per_point = 5;
        double *local_buffer = (double *)malloc(local_count * vals_per_point * sizeof(double));

        for (unit_j = start; unit_j < end; unit_j++) {
            memset(Tdiag, 0, m * sizeof(double));
            memset(Toffdiag, 0, (m - 1) * sizeof(double));
            
            clock_t start_time = clock();
            double tmp = lanczos_quadrature(pSPARC, pRPA, Tdiag, Toffdiag, m, unit_j, MPI_COMM_SELF, omegaIndex);
            clock_t end_time = clock();
            double diff = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
            
            result += tmp;

            int i, j, k;
            from_unit(unit_j, pSPARC->Nx_d, pSPARC->Ny_d, pSPARC->Nz_d, &i, &j, &k);

            int offset = (unit_j - start) * vals_per_point;
            local_buffer[offset + 0] = (double)i;
            local_buffer[offset + 1] = (double)j;
            local_buffer[offset + 2] = (double)k;
            local_buffer[offset + 3] = tmp;
            local_buffer[offset + 4] = diff;
        }

        double *global_buffer = NULL;
        int *recvcounts = NULL;
        int *displs = NULL;

        if (rank == 0) {
            global_buffer = (double *)malloc(pSPARC->Nd * vals_per_point * sizeof(double));
            recvcounts = (int *)malloc(size * sizeof(int));
            displs = (int *)malloc(size * sizeof(int));

            for (int r = 0; r < size; r++) {
                int r_start, r_end;
                if (r < remainder) {
                    r_start = r * (chunk_size + 1);
                    r_end = r_start + (chunk_size + 1);
                } else {
                    r_start = r * chunk_size + remainder;
                    r_end = r_start + chunk_size;
                }
                recvcounts[r] = (r_end - r_start) * vals_per_point;
                displs[r] = r_start * vals_per_point;
            }
        }

        MPI_Gatherv(local_buffer, local_count * vals_per_point, MPI_DOUBLE,  global_buffer, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            char fname[100];
            sprintf(fname, "diags_and_times_%d.txt", omegaIndex);
            FILE *fp = fopen(fname, "w");
            for (int p = 0; p < pSPARC->Nd; p++) {
                int idx = p * vals_per_point;
                fprintf(fp, "(%d,%d,%d): %e %e\n", 
                        (int)global_buffer[idx+0], (int)global_buffer[idx+1], (int)global_buffer[idx+2], 
                        global_buffer[idx+3], global_buffer[idx+4]);
            }
            fclose(fp);
            free(global_buffer);
            free(recvcounts);
            free(displs);
        }

        free(local_buffer);
    }
    else{
        for (unit_j=start;unit_j<end;unit_j++) {
            memset(Tdiag,0,m*sizeof(double));
            memset(Toffdiag,0,(m-1)*sizeof(double));
            double tmp = lanczos_quadrature(pSPARC,pRPA,Tdiag,Toffdiag,m,unit_j,MPI_COMM_SELF,omegaIndex);
            result+=tmp;
        }
    }
    if(size>1){
        MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    }

    free(Tdiag); free(Toffdiag); 

    return result;
}


void fourier_interp_1d(const double *src, int src_len, double *dst, int dst_len) {
    int K = src_len / 2;
    double complex *X = (double complex *)malloc((K + 1) * sizeof(double complex));
    for (int k = 0; k <= K; k++) {
        double complex sum = 0;
        for (int n = 0; n < src_len; n++) {
            double angle = -2.0 * M_PI * k * n / src_len;
            sum += src[n] * (cos(angle) + I * sin(angle));
        }
        X[k] = sum;
    }
    for (int m = 0; m < dst_len; m++) {
        double complex val = X[0];
        for (int k = 1; k < (src_len + 1) / 2; k++) {
            double angle = 2.0 * M_PI * k * m / dst_len;
            double complex phase = cos(angle) + I * sin(angle);
            val += 2.0 * creal(X[k] * phase);
        }
        if (src_len % 2 == 0) {
            double angle_nyq = 2.0 * M_PI * K * m / dst_len;
            val += creal(X[K]) * cos(angle_nyq);
        }
        dst[m] = creal(val) / src_len;
    }

    free(X);
}

double interp_lanczos_quadrature_wrapper(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int n, int SKIP, int m, int omegaIndex) {
    int Nx=pSPARC->Nx_d, Ny=pSPARC->Ny_d, Nz=pSPARC->Nz_d;
    
    int cNx=Nx / SKIP; 
    int cNy=Ny / SKIP;
    int cNz=Nz / SKIP;
    int total_coarse=cNx * cNy * cNz;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *coarse_data=(double *)calloc(total_coarse, sizeof(double));
    double *Tdiag=(double *)malloc(m * sizeof(double));
    double *Toffdiag=(double *)malloc((m - 1) * sizeof(double));

    for (int idx=rank; idx < total_coarse; idx += size) {
        int ck=idx / (cNx * cNy);
        int cj=(idx / cNx) % cNy;
        int ci=idx % cNx;

        int unit_j=(ck * SKIP * Nx * Ny) + (cj * SKIP * Nx) + (ci * SKIP);

        memset(Tdiag, 0, m * sizeof(double));
        memset(Toffdiag, 0, (m - 1) * sizeof(double));
        
        coarse_data[idx]=lanczos_quadrature(pSPARC, pRPA, Tdiag, Toffdiag, m, unit_j, MPI_COMM_SELF, omegaIndex);
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, coarse_data, total_coarse, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(coarse_data, NULL, total_coarse, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    double total_result=0.0;

    if (rank == 0) {
        double *tmp_X = (double *)malloc(sizeof(double) * Nx * cNy * cNz);
        for (int kz = 0; kz < cNz; kz++) {
            for (int ky = 0; ky < cNy; ky++) {
                int src_offset = (kz * cNy + ky) * cNx;
                int dst_offset = (kz * cNy + ky) * Nx;
                fourier_interp_1d(&coarse_data[src_offset], cNx, &tmp_X[dst_offset], Nx);
            }
        }

        double *tmp_Y = (double *)malloc(sizeof(double) * Nx * Ny * cNz);
        double *src_col = (double *)malloc(sizeof(double) * cNy);
        double *dst_col = (double *)malloc(sizeof(double) * Ny);

        for (int kz = 0; kz < cNz; kz++) {
            for (int kx = 0; kx < Nx; kx++) {
                for (int ky = 0; ky < cNy; ky++) {
                    src_col[ky] = tmp_X[(kz * cNy + ky) * Nx + kx];
                }
                fourier_interp_1d(src_col, cNy, dst_col, Ny);
                for (int ky = 0; ky < Ny; ky++) {
                    tmp_Y[(kz * Ny + ky) * Nx + kx] = dst_col[ky];
                }
            }
        }

        double *fine_data = (double *)malloc(sizeof(double) * Nx * Ny * Nz);
        double *src_depth = (double *)malloc(sizeof(double) * cNz);
        double *dst_depth = (double *)malloc(sizeof(double) * Nz);

        for (int ky = 0; ky < Ny; ky++) {
            for (int kx = 0; kx < Nx; kx++) {
                for (int kz = 0; kz < cNz; kz++) {
                    src_depth[kz] = tmp_Y[(kz * Ny + ky) * Nx + kx];
                }
                fourier_interp_1d(src_depth, cNz, dst_depth, Nz);
                for (int kz = 0; kz < Nz; kz++) {
                    fine_data[(kz * Ny + ky) * Nx + kx] = dst_depth[kz];
                }
            }
        }

        for (int i=0; i < (Nx * Ny * Nz); i++) {
            total_result += fine_data[i];
        }

        free(tmp_X);
        free(tmp_Y);
        free(fine_data);
        free(src_col);
        free(dst_col);
        free(src_depth);
        free(dst_depth);
    }
    free(coarse_data);
    free(Tdiag);
    free(Toffdiag);

    return total_result;
}

void lanczos_quadrature_RPA(SPARC_OBJ *pSPARC, RPA_OBJ *pRPA, int qptIndex, int omegaIndex) {

    double qptOmegaWeight=pRPA->qptWts[qptIndex] / pSPARC->Nkpts_sym * pRPA->omegaWts[omegaIndex];
    double ErpaTerm=0.0;
    double start_time,end_time;
    
    MPI_Comm nuChi0Eigscomm=pRPA->nuChi0Eigscomm;
    int nuChi0EigscommIndex=pRPA->nuChi0EigscommIndex; 
    int rank, globalrank;
    MPI_Comm_rank(nuChi0Eigscomm, &rank);
    if((!pRPA->nuChi0EigscommIndex) &&(!rank)) {
        FILE *output_fp=fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"***************************************************************************\n");
        fprintf(output_fp,"q-point %d (reduced coords %.3f %.3f %.3f), weight %.3f\n omega %d (value %.3f, 0~1 value %.3f, weight %.3f)\n",
            qptIndex + 1, pRPA->q1[qptIndex]*pSPARC->range_x/(2*M_PI), pRPA->q2[qptIndex]*pSPARC->range_y/(2*M_PI), pRPA->q3[qptIndex]*pSPARC->range_z/(2*M_PI), pRPA->qptWts[qptIndex],
            omegaIndex + 1, pRPA->omega[omegaIndex], pRPA->omega01[omegaIndex], pRPA->omegaWts[omegaIndex]);
        fprintf(output_fp,"nIter|ErpaTerm(Ha/atom)|Timing (s)\n");
        fclose(output_fp);
    }
    int rankHpDiagComm;
    MPI_Comm_rank(pRPA->HpDiagComm, &rankHpDiagComm);
    MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);

    int flagNoDmcomm=(pSPARC->spincomm_index < 0 || pSPARC->kptcomm_index < 0 || pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL);
    
    if ((nuChi0EigscommIndex != -1) && pRPA->flagCOCGinitial && (!flagNoDmcomm)) {
        if (pSPARC->isGammaPoint) {
            collect_allXorb_allLambdas_gamma(pSPARC, pRPA);
        } else {
            collect_allXorb_allLambdas_kpt(pSPARC, pRPA);
            send_recv_allXorb_allLambdas_kPq(pSPARC, pRPA, qptIndex);
        }
    }

    start_time=MPI_Wtime();
    int num_RHS=pRPA->flagLQ;


    if(pRPA->flagINTERP_MODE){
        if(pRPA->flagLQ==1){
            ErpaTerm=interp_lanczos_quadrature_wrapper(pSPARC, pRPA,pSPARC->Nd,pRPA->flagNUM_COARSE_SKIP,pRPA->flagNUM_LQ_ITER,omegaIndex);
        }
        else{
            //HAQUE TODO: Include interpolation within block 
        }
    }
    else{
        if(pRPA->flagLQ==1){
            ErpaTerm=lanczos_quadrature_wrapper(pSPARC, pRPA,pSPARC->Nd,pRPA->flagNUM_LQ_ITER,omegaIndex);
        }
        else{
            //HAQUE TODO: Include block 
        }
    }
    end_time=MPI_Wtime();

    if (!globalrank) {
        ErpaTerm*= qptOmegaWeight / (pRPA->omega01[omegaIndex]*pRPA->omega01[omegaIndex]) / (2.0*M_PI);
        printf("qpt %d, omega %d, subspace iteration %d, ErpaTerm %.6f, lastErpaTerm %.6f, tolErpaTermConverge %.6f, flagIter %d\n", qptIndex + 1, omegaIndex + 1, 0.0, ErpaTerm, 0.0, 0.0, 0);
        FILE *output_fp=fopen(pRPA->filename_out,"a");
        fprintf(output_fp,"% 3d   %.8E   %.2f\n", 1, ErpaTerm / (double)pSPARC->n_atom, end_time-start_time);
        fclose(output_fp);
        pRPA->ErpaTerms[qptIndex*pRPA->Nomega + omegaIndex]=ErpaTerm;
    }

}
