#include "common.h"
#include "omp.h"


void CSR_SpMV(int            m,
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
    printf("\n------------------------CSR_SpMV------------------------\n");


    /* preprocessing */
    double t1 = omp_get_wtime();

    // cout << "finishing 0" << endl << flush;

    int threads = omp_get_max_threads();
    int *row_start = new int[threads]();
    int *row_end = new int[threads]();
    row_start[0] = 0;
    for(int i = 0, j = 0; i < m; i++)
    {
        // cout << i << endl << flush;
        // cout << csrRowPtrA[i+1] << endl << flush;
        if (csrRowPtrA[i+1]-csrRowPtrA[row_start[j]] > nnzA / threads)
        {
            row_end[j] = row_start[j+1] = i+1;
            // cout << row_start[j] << " " << row_end[j] << endl;
            j++;
        }
        // cout << i << endl << flush;
    }
    row_end[threads-1] = m;

    // cout << "finishing 1" << endl << flush;


    double t2 = omp_get_wtime();
    *pre_time_add = (t2 - t1) * 1000.0;


    /* calculate Ax=y */
    double time_solve = 0;
    for (int ii = 0; ii < BENCH_REPEAT; ii++)
    {
        t1 = omp_get_wtime();


#pragma omp parallel
        {
            int id = omp_get_thread_num();
            for (int i = row_start[id]; i < row_end[id]; i++)
            {
                VALUE_TYPE sum = 0;
                for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++) 
                    sum += x[csrColIdxA[j]] * csrValA[j];
                y[i] = sum;
            }
        }

        
        t2 = omp_get_wtime();
        time_solve += t2 - t1;
    }

    // cout << "finishing 2" << endl << flush;

    double dataSize = (double)((m+1)*sizeof(int) + nnzA*sizeof(int) + nnzA*sizeof(VALUE_TYPE) + 2*m*sizeof(VALUE_TYPE));

    time_solve = time_solve * 1000.0 / BENCH_REPEAT;
    *solve_time_add = time_solve;
    *gflops_add = 2 * nnzA / (1e6 * time_solve);
    *bandwith_add = dataSize / (1e6 * time_solve);


    // cout << "finishing 3" << endl << flush;

    delete[] row_start;
    delete[] row_end;
}