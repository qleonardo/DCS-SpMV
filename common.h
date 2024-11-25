#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>

#include <sys/time.h>
#include "DC.h"
#include "omp.h"

using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif


#ifndef BENCH_REPEAT
#define BENCH_REPEAT 100
#endif



#ifndef CHECK_CORRECTNESS
#define CHECK_CORRECTNESS

bool CheckCorrectness(int m, VALUE_TYPE *y_ref, VALUE_TYPE *y)
{/**************************************CheckCorrectness******************************************/
    
    int error_count = 0;
    for (int i = 0; i < m; i++)
        //both relative error and absolute error should be considered.
        if (fabs(y_ref[i] - y[i]) > 1e-5 * max(fabs(y_ref[i]), fabs(y[i])) && fabs(y_ref[i] - y[i]) > 1e-5)
        {
            error_count++;
            // cout << "ROW [ " << i << " ], NNZ SPAN: "
            //         << "\t ref = " << y[i]
            //         << ", \t std = " << y_ref[i]
            //     << ", \t error = " << fabs(y_ref[i] - y[i])
            //         << endl << flush;
        //    break;
        }

    if (error_count == 0)
    {
        cout << "Check... PASS! \n" << flush;
        return 1;
    }
    else
        return 0;
}

#endif



#ifndef REORDERING
#define REORDERING

void sssRowReordering(int *RowRev, int *RowPerm, int *new_sssRowPtrA, int *new_sssColIdxA, VALUE_TYPE *new_sssDValA, VALUE_TYPE *new_sssValA, VALUE_TYPE *new_x, int *sssRowPtrA, 
                                                                                            int *sssColIdxA, VALUE_TYPE *sssDValA, VALUE_TYPE *sssValA, VALUE_TYPE *x, int m)
{
    int j = 0;
    for (int i = 0; i < m; i++)
    {
        int dst = RowRev[i];
        new_sssRowPtrA[i] = j;
        j += sssRowPtrA[dst + 1]-sssRowPtrA[dst];
    }
    new_sssRowPtrA[m] = sssRowPtrA[m];

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        int dst = RowRev[i];
        int row_start = sssRowPtrA[dst];
        int row_end = sssRowPtrA[dst + 1];
        int tot = row_end -row_start;
        new_sssDValA[i] = sssDValA[dst];
        new_x[i] = x[dst];
        pair<int,VALUE_TYPE> *pairs = new pair<int,VALUE_TYPE>[tot];
        for (int k = row_start; k < row_end; k++)
        {
            pairs[k-row_start].first = RowPerm[sssColIdxA[k]];
            pairs[k-row_start].second = sssValA[k];
        }
        row_start = new_sssRowPtrA[i];
        sort(pairs, pairs+tot);
        for (int k = 0; k < tot; k++)
        {
            new_sssColIdxA[row_start+k] = pairs[k].first;
            new_sssValA[row_start+k] = pairs[k].second;
        }
        delete[] pairs;
    }
}

#endif



#ifndef CONFLICT_GRAPH
#define CONFLICT_GRAPH

long long create_Conflict_Graph(int* sssRowPtrA, int* sssColIdxA, int *nRowPerRow, int **Row2Row, int m)
{
    double t1 = omp_get_wtime();

    // cout << "create conflict graph 0" << endl << flush;
    int threads = omp_get_max_threads();
    vector<int> *output[threads];
    int **vis = new int*[threads];
    
#pragma omp parallel for
    for(int i = 0; i < threads; i++)
    {
        vis[i] = new int[m];
        output[i] = new vector<int>[m]();
        for(int j = 0; j < m; j++)
            vis[i][j] = -1;
    }

    // cout << "create conflict graph 1" << endl << flush;
#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < m; i++)
    {
        int id = omp_get_thread_num();
        for(int j = sssRowPtrA[i]; j < sssRowPtrA[i+1]; j++)
            output[id][sssColIdxA[j]].push_back(i);
    }


    // cout << "create conflict graph 2" << endl << flush;
    long long conflicts = 0;
#pragma omp parallel for reduction(+:conflicts)
    for(int i = 0; i < m; i++)
    {
        int sum = 0;
        for(int t = 0; t < threads; t++)
            sum += output[t][i].size();
        conflicts += 1ll*(sum-1)*sum/2;
    }
    if (conflicts > 1e10) return conflicts;
    conflicts = 0;
            
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
                if (vis[id][row] != i)
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
                    if (vis[id][row] != i)
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

    for(int i = 0; i < threads; i++)
    {
        for(int j = 0; j < m; j++)
            output[i][j].clear();
        delete[] output[i];
        delete[] vis[i];
    }
    delete[] vis;
    
    double t2 = omp_get_wtime();
    // cout << "create conflict graph time: " << t2 - t1 << endl << flush;


    return conflicts;
}

#endif



#ifndef CREATE_TREE
#define CREATE_TREE
int CreateTree(DC *DCRoot, int **Row2Row, int *nRowPerRow, int *sssRowPtrA, int m)
{
    int *nRowPerRow_backup = new int[m];
    int **Row2Row_backup = new int*[m];
    int *RowValue = new int[m];

#pragma omp parallel for
    for(int i = 0; i < m; i++)
    {
        RowValue[i] = (sssRowPtrA[i+1] - sssRowPtrA[i]) * 2 + 1;
        nRowPerRow_backup[i] = nRowPerRow[i];
        Row2Row_backup[i] = new int[nRowPerRow_backup[i]];
        for(int j = 0; j < nRowPerRow[i]; j++)
            Row2Row_backup[i][j] = Row2Row[i][j];
    }

    int res = DCRoot->DC_creation(Row2Row_backup, nRowPerRow_backup, RowValue, m);

#pragma omp parallel for
    for(int i = 0; i < m; i++)
        delete[] Row2Row_backup[i];
    delete[] Row2Row_backup;
    delete[] nRowPerRow_backup;
    delete[] RowValue;

    return res;
}

#endif


#ifndef SSS_TO_CSR
#define SSS_TO_CSR
void convert_SSS_to_CSR(int m, int *sssRowPtrA, int *sssColIdxA, VALUE_TYPE *sssValA, VALUE_TYPE *sssDValA, int *&csrRowPtrA, int *&csrColIdxA, VALUE_TYPE *&csrValA, int start_CSR)
{

    int threads = omp_get_max_threads();
    vector<pair<int,VALUE_TYPE> > *val[threads];
#pragma omp parallel for
    for(int i = 0; i < threads; i++)
        val[i] = new vector<pair<int,VALUE_TYPE> >[m];
#pragma omp parallel for
    for(int i = start_CSR; i < m; i++)
    {
        int id = omp_get_thread_num(); 
        for(int j = sssRowPtrA[i]; j < sssRowPtrA[i+1]; j++)
        {
            val[id][i].push_back(make_pair(sssColIdxA[j], sssValA[j]));
            val[id][sssColIdxA[j]].push_back(make_pair(i, sssValA[j]));
        }
        if (fabs(sssDValA[i])>1e-6)
            val[id][i].push_back(make_pair(i, sssDValA[i]));
    }

    int nnz = 0;
    vector<pair<int, VALUE_TYPE> > *Val = new vector<pair<int, VALUE_TYPE> >[m];
#pragma omp parallel for reduction(+:nnz)
    for (int i = 0; i < m; i++)
    {
        for(int j = 0; j < threads; j++)
        {
            Val[i].insert(Val[i].end(), val[j][i].begin(), val[j][i].end());
            val[j][i].clear();
        }
        sort(Val[i].begin(), Val[i].end());
        nnz += Val[i].size();
    }
#pragma omp parallel for
    for(int j = 0; j < threads; j++)
        delete[] val[j];

    
    int tot = 0;
    csrRowPtrA = new int[m+1];
    csrColIdxA = new int[nnz];
    csrValA = new VALUE_TYPE[nnz];
    for (int i = 0; i < m; i++)
    {
        csrRowPtrA[i] = tot;
        tot += Val[i].size();
    }
    csrRowPtrA[m] = tot;
    
// #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        int offset = csrRowPtrA[i];
        for(int j = 0; j < Val[i].size(); j++)
        {
            csrColIdxA[offset+j] = Val[i][j].first;
            csrValA[offset+j] = Val[i][j].second;
            // cout << i << " " << Val[i][j].first << " " << Val[i][j].second << endl;
        }
        Val[i].clear();
    }
    delete[] Val;
}
#endif