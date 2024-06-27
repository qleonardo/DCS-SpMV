#include "common.h"
#include "DC.h"


inline void DC_Hybrid_SymSpMV_kernel(char ** userArgs, DCArgs *treeArgs)
{
    int *sssRowPtrA         = (int *)userArgs[0];
    int *sssColIdxA         = (int *)userArgs[1];
    VALUE_TYPE *sssValA     = (VALUE_TYPE *)userArgs[2];
    VALUE_TYPE *sssDValA    = (VALUE_TYPE *)userArgs[3];
    VALUE_TYPE *x           = (VALUE_TYPE *)userArgs[4];
    VALUE_TYPE *y_ref       = (VALUE_TYPE *)userArgs[5];
    int *csrRowPtrA         = (int *)userArgs[6];
    int *csrColIdxA         = (int *)userArgs[7];
    VALUE_TYPE *csrValA     = (VALUE_TYPE *)userArgs[8];

    int ns = treeArgs->firstRow;
    int ne = treeArgs->lastRow;

    if (ns >= ne) return;
    

    for (int i = ns; i < ne; i++)
    {
        VALUE_TYPE sum = sssDValA[i] * x[i];
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++) 
            sum += x[csrColIdxA[j]] * csrValA[j];
        for (int j = sssRowPtrA[i]; j < sssRowPtrA[i+1]; j++)
        {
            int col = sssColIdxA[j];
            VALUE_TYPE val = sssValA[j];
            sum += x[col] * val;
            y_ref[col] += x[i] * val;
        }
        y_ref[i] += sum;
    }
}



double DC_Hybrid_SymSpMV(int            m,
                            int             nnzA,
                            int             *sssRowPtrA,
                            int             *sssColIdxA,
                            VALUE_TYPE      *sssValA,
                            VALUE_TYPE      *sssDValA,
                            VALUE_TYPE      *&x,
                            VALUE_TYPE      *&y,
                            int             *nRowPerRow,
                            int             **Row2Row,
                            double          *solve_time_add,
                            double          *gflops_add,
                            double          *bandwith_add,
                            double          *pre_time_add,
                            int             nbParts,
                            int             Level)
{
    /* preprocessing */

    double t1 = omp_get_wtime();


    /* create DC Tree */
    int start_CSR;                                  // for DCS Hybrid SpMV
    DC *DCRoot = new DC(m, nbParts, Level);
    start_CSR = CreateTree(DCRoot, Row2Row, nRowPerRow, sssRowPtrA, m);                         


    int *RowRev = DCRoot->DC_get_RowRev();
    int *RowPerm = DCRoot->DC_get_RowPerm();
    int *new_sssRowPtrA = new int[m+1];
    int *new_sssColIdxA = new int[nnzA/2];
    VALUE_TYPE *new_sssValA = new VALUE_TYPE[nnzA/2];
    VALUE_TYPE *new_sssDValA = new VALUE_TYPE[m];
    VALUE_TYPE *new_x = new VALUE_TYPE[m];


    /* matrix reordering */
    sssRowReordering(RowRev, RowPerm, new_sssRowPtrA, new_sssColIdxA, new_sssDValA, new_sssValA, new_x, sssRowPtrA, sssColIdxA, sssDValA, sssValA, x, m);


    /* We use the naive SpMV (ignoring symmetry) to calculate the rows from start_CSR to m-1 */
    sssRowReordering(RowRev, RowPerm, new_sssRowPtrA, new_sssColIdxA, new_sssDValA, new_sssValA, new_x, sssRowPtrA, sssColIdxA, sssDValA, sssValA, x, m);


    int *csrRowPtrA, *csrColIdxA;
    VALUE_TYPE *csrValA;
    convert_SSS_to_CSR(m, new_sssRowPtrA, new_sssColIdxA, new_sssValA, new_sssDValA, csrRowPtrA, csrColIdxA, csrValA, start_CSR);

    
    // cout << endl << start_CSR << endl;
    // cout << tot - csrRowPtrA[start_CSR] << " " << nnz << " " << tot << endl;

    VALUE_TYPE *y_ref = new VALUE_TYPE[m];

    double t2 = omp_get_wtime();
    *pre_time_add = (t2 - t1) * 1000;
    


    /* calculate Ax=y */
    double time_solve = 0;
    for (int ii = 0; ii < BENCH_REPEAT; ii++)
    {
        memset(y_ref, 0, m*sizeof(VALUE_TYPE));

        t1 = omp_get_wtime();


        char *userArgs[9] = {(char *)new_sssRowPtrA, (char *)new_sssColIdxA, (char *)new_sssValA, (char *)new_sssDValA, (char *)new_x, (char *)y_ref, (char *)csrRowPtrA, (char *)csrColIdxA, (char *)csrValA};
        DCRoot->DC_traversal(DC_Hybrid_SymSpMV_kernel, userArgs, Level);
        

#pragma omp parallel for 
        for(int i = start_CSR; i < m; i++)
        {
            VALUE_TYPE sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++) 
                sum += new_x[csrColIdxA[j]] * csrValA[j];
            y_ref[i] += sum;
        }


        t2 = omp_get_wtime();
        time_solve += t2 - t1;
    }


    double dataSize = (double)((m+1)*sizeof(int) + nnzA*sizeof(int) + nnzA*sizeof(VALUE_TYPE) + 2*m*sizeof(VALUE_TYPE));

    time_solve = time_solve * 1000.0 / BENCH_REPEAT;
    *solve_time_add = time_solve;
    *gflops_add = 2 * nnzA / (1e6 * time_solve);
    *bandwith_add = dataSize / (1e6 * time_solve);


#pragma omp parallel for
    for(int i = 0; i < m; i++)
        y[i] = y_ref[RowPerm[i]];

    delete[] y_ref;
    delete[] new_sssRowPtrA;
    delete[] new_sssColIdxA;
    delete[] new_sssDValA;
    delete[] new_sssValA;
    delete[] csrRowPtrA;
    delete[] csrColIdxA;
    delete[] csrValA;
    delete[] new_x;
    delete DCRoot;
}
