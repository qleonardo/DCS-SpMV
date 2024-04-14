#include "common.h"


void create_conflict_graph_CFH(int m, int *sssRowPtrA, int *sssColIdxA, int *RowPerm, int * RowRev)
{
    double t1 = omp_get_wtime();

    int BLK = 8;
    // cout << "create conflict graph 0" << endl << flush;
    int threads = omp_get_max_threads();
    int num_block = (m + BLK - 1) / BLK;
    vector<int> *output[threads];
    int **vis = new int*[threads];
    
#pragma omp parallel for
    for(int i = 0; i < threads; i++)
    {
        vis[i] = new int[num_block];
        output[i] = new vector<int>[m];
        for(int j = 0; j < num_block; j++)
            vis[i][j] = -1;
    }

    // cout << "create conflict graph 1" << endl << flush;
#pragma omp parallel for schedule(dynamic)
    for(int ii = 0; ii < num_block; ii++)
    {
        int id = omp_get_thread_num();
        for (int i = ii*BLK; i < (ii+1)*BLK && i < m; i++)
            for(int j = sssRowPtrA[i]; j < sssRowPtrA[i+1]; j++)
                output[id][sssColIdxA[j]].push_back(ii);
    }

    
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < m; i++)
    {
        for(int t = 0; t < threads; t++)
        {
            sort(output[t][i].begin(), output[t][i].end());
            auto it = unique(output[t][i].begin(), output[t][i].end());
            output[t][i].erase(it);
        }
    }


    // cout << "create conflict graph 2" << endl << flush;
    long long conflicts = 0;

    // cout << "create conflict graph 3" << endl << flush;
#pragma omp parallel for
    for(int i = 0; i < threads; i++)
        for(int j = 0; j < num_block; j++)
            vis[i][j] = -1;
            
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < m; i++)
    {
        vector<int> confRow;
        int id = omp_get_thread_num();

        for(int j = sssRowPtrA[i]; j < sssRowPtrA[i+1]; j++)
        {
            confRow.push_back(sssColIdxA[j]);
            vis[id][sssColIdxA[j]] = i;
        }

        for(int t = 0; t < threads; t++)
            for (int j = 0; j < output[t][i].size(); j++)
            {
                int row = output[t][i][j];
                if (vis[id][row] != i && row <= i)
                {
                    confRow.push_back(row);
                    vis[id][row] = i;
                }
            }

        for(int t = 0; t < threads; t++) 
            for(int j = sssRowPtrA[i]; j < sssRowPtrA[i+1]; j++)
            {
                int jj = sssColIdxA[j];
                for (int k = 0; k < output[t][jj].size(); k++)
                {
                    int row = output[t][jj][k];
                    if (vis[id][row] != i && row <= i)
                    {
                        confRow.push_back(row);
                        vis[id][row] = i;
                    }
                }

            }

        int k = 0;
        Row2Row[i] = new int[confRow.size()];
        for(int j = 0; j < confRow.size(); j++)
            if (confRow[j] != i)
                Row2Row[i][k++] = confRow[j];
        nRowPerRow[i] = k;
        // cout << k << endl << flush;

        confRow.clear();
    }

    for(int i = 0; i < m; i++)
        conflicts += nRowPerRow[i];
    // cout << "create conflict graph 4" << endl << flush;
#pragma omp parallel for
    for(int i = 0; i < threads; i++)
    {
        for(int j = 0; j < m; j++)
            output[i][j].clear();
        delete[] output[i];
        delete[] vis[i];
    }
    delete[] vis;
    
    double t2 = omp_get_wtime();
    cout << "create conflict graph time: " << t2 - t1 << endl << flush;


    return conflicts;
}


double CFH_SymSpMV(int            m,
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
                            double          *pre_time_add)
{
    printf("\n------------------------CFH_SymSpMV------------------------\n");


    /* preprocessing */

    double t1 = omp_get_wtime();


    int *RowPerm, *RowRev;
    create_conflict_graph_CFH(m, sssRowPtrA, sssColIdxA, RowPerm, RowRev);


    double t2 = omp_get_wtime();
    *pre_time_add = (t2 - t1) * 1000;
    

    /* calculate Ax=y */
    double time_solve = 0;
    for (int ii = 0; ii < BENCH_REPEAT; ii++)
    {
        memset(y_ref, 0, m*sizeof(VALUE_TYPE));

        t1 = omp_get_wtime();


       


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


}
