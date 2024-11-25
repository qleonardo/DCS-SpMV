#include <mkl_spblas.h>
#include "common.h"
#include "omp.h"

void MKL_SpMV(int            m,
                 int            nnzA,
                 int            *&csrRowPtrA,
                 int            *&csrColIdxA,
                 VALUE_TYPE     *&csrValA,
                 VALUE_TYPE     *&x,
                 VALUE_TYPE     *&y,
                 double         *solve_time_add,
                 double         *gflops_add,
                 double         *bandwith_add,
                 double         *pre_time_add)
{
    // printf("\n------------------------MKL_SpMV------------------------\n");


    /* preprocessing */
    double t1 = omp_get_wtime();


    int *row_start = new int[m];
    int *row_end = new int[m];
#pragma omp parallel for
    for(int i = 0; i < m; i++)
    {
        row_start[i] = csrRowPtrA[i];
        row_end[i] = csrRowPtrA[i+1];
    }

    sparse_matrix_t A = NULL;
    sparse_status_t res = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, m, m, row_start, row_end, csrColIdxA, csrValA);

    matrix_descr dsc;
    dsc.type = SPARSE_MATRIX_TYPE_GENERAL;


    double t2 = omp_get_wtime();
    *pre_time_add = (t2 - t1) * 1000;


    /* calculate Ax=y */
    double time_solve = 0;
    for (int ii = 0; ii < BENCH_REPEAT; ii++)
    {
        memset(y, 0, m*sizeof(VALUE_TYPE));
        
        t1 = omp_get_wtime();


        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, A, dsc, x, 0, y);


        t2 = omp_get_wtime();
        time_solve += t2 - t1;
    }
    
    
    double dataSize = (double)((m+1)*sizeof(int) + nnzA*sizeof(int) + nnzA*sizeof(VALUE_TYPE) + 2*m*sizeof(VALUE_TYPE));

    time_solve = time_solve * 1000.0 / BENCH_REPEAT;
    *solve_time_add = time_solve;
    *gflops_add = 2 * nnzA / (1e6 * time_solve);
    *bandwith_add = dataSize / (1e6 * time_solve);


    delete[] row_start;
    delete[] row_end;
    
    mkl_sparse_destroy(A);
}