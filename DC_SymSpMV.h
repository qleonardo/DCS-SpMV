#include "common.h"
#include "DC.h"
#include "omp.h"

void DC_SymSpMV_kernel(char ** userArgs, DCArgs *treeArgs)
{
    int *sssRowPtrA         = (int *)userArgs[0];
    int *sssColIdxA         = (int *)userArgs[1];
    VALUE_TYPE *sssValA     = (VALUE_TYPE *)userArgs[2];
    VALUE_TYPE *sssDValA    = (VALUE_TYPE *)userArgs[3];
    VALUE_TYPE *x           = (VALUE_TYPE *)userArgs[4];
    VALUE_TYPE *y_ref       = (VALUE_TYPE *)userArgs[5];

    int ns = treeArgs->firstRow;
    int ne = treeArgs->lastRow;
// #pragma omp critical
//     cout << ns << " " << ne << endl << flush;

    if (ns >= ne) return;

    for (int i = ns; i < ne; i++)
    {
        VALUE_TYPE sum = sssDValA[i] * x[i];
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



double DC_SymSpMV(int            m,
                            int             nnzA,
                            int             *&sssRowPtrA,
                            int             *&sssColIdxA,
                            VALUE_TYPE      *&sssValA,
                            VALUE_TYPE      *&sssDValA,
                            VALUE_TYPE      *&x,
                            VALUE_TYPE      *&y,
                            double          *solve_time_add,
                            double          *gflops_add,
                            double          *bandwith_add,
                            double          *pre_time_add,
                            int             nbParts,
                            int             Level)
{
    printf("\n------------------------DC_SymSpMV------------------------\n");


    /* preprocessing */

    double t1 = omp_get_wtime();


    int *nRowPerRow = new int[m];
    int **Row2Row = new int*[m];
    long long conflicts = create_Conflict_Graph(sssRowPtrA, sssColIdxA, nRowPerRow, Row2Row, m);


    if (conflicts >= 2e9)
    {
        cout << conflicts << " " << 2500000000 << " " << 2500000000 << " " << 2500000000 << endl << flush;
        cout << 1e9 << "\t" << 1e9 << "\t" << 0 << endl << endl;
        return 0;
    }
    int maxConflict = -2e9, minConflict = 2e9;
    double conflict_average = conflicts * 1.0 / m;
    double conflict_Std = 0.0;

    for (int i = 0; i < m; i++)
    {
        int nnzConflict = nRowPerRow[i];
        conflict_Std += (nnzConflict-conflict_average)*(nnzConflict-conflict_average);
        maxConflict = max(maxConflict, nnzConflict);
        minConflict = min(minConflict, nnzConflict);
    }
    conflict_Std /= m;
    conflict_Std = sqrt(conflict_Std);
    cout << conflict_average << " " << maxConflict << " " << minConflict << " " << conflict_Std << endl << flush;

    DC *DCRoot = new DC(m, nbParts, Level);
    CreateTree(DCRoot, Row2Row, nRowPerRow, sssRowPtrA, m);

    int *RowRev = DCRoot->DC_get_RowRev();
    int *RowPerm = DCRoot->DC_get_RowPerm();
    int *new_sssRowPtrA = (int *)malloc((m+1) * sizeof(int));
    int *new_sssColIdxA = (int *)malloc(nnzA/2 * sizeof(int));
    VALUE_TYPE *new_sssValA = (VALUE_TYPE *)malloc(nnzA/2 * sizeof(VALUE_TYPE));
    VALUE_TYPE *new_sssDValA = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
    VALUE_TYPE *new_x = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

    sssRowReordering(RowRev, RowPerm, new_sssRowPtrA, new_sssColIdxA, new_sssDValA, new_sssValA, new_x, sssRowPtrA, sssColIdxA, sssDValA, sssValA, x, m);

    VALUE_TYPE *y_ref = new VALUE_TYPE[m];


    double t2 = omp_get_wtime();
    *pre_time_add = (t2 - t1) * 1000;
    

    /* calculate Ax=y */
    double time_solve = 0;
    for (int ii = 0; ii < BENCH_REPEAT; ii++)
    {
        memset(y_ref, 0, m*sizeof(VALUE_TYPE));

        t1 = omp_get_wtime();

        char *userArgs[9] = {(char *)new_sssRowPtrA, (char *)new_sssColIdxA, (char *)new_sssValA, (char *)new_sssDValA, (char *)new_x, (char *)y_ref};
        DCRoot->DC_traversal(DC_SymSpMV_kernel, userArgs);


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
    delete[] new_x;
    delete DCRoot;
}
