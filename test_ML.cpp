#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "read_mtx.h"
#include "CSR_SpMV.h"
#include "DC.h"
#include "DC_SymSpMV.h"
#include "DC_Hybrid_SymSpMV.h"
using namespace std;


int main(int argc, char ** argv)
{
    int m, n, nnzA, isSymmetricA;
    int* csrRowPtrA;
    int* csrColIdxA;
    VALUE_TYPE* csrValA;
    
    int argi = 1;
    int threads = omp_get_max_threads();

    int predict, truth;
    char  *filename, *matrix_name;
    if(argc > argi)
    {
        filename = argv[argi++];
        matrix_name = argv[argi++];
        predict = atoi(argv[argi++]);
        truth = atoi(argv[argi++]);
    }

    srand(time(NULL));
    read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);

    ofstream ml_result;
    ml_result.open("ml_result.csv", ios::app);
    ml_result << matrix_name << ",";


    VALUE_TYPE *x = new VALUE_TYPE[m];
    VALUE_TYPE *y = new VALUE_TYPE[m];
    VALUE_TYPE *y_ref = new VALUE_TYPE[m];
    for (int i = 0; i < m; i++) 
        x[i] = (rand() % 200 - 100) / 100.0;



    double CSR_solve_time, CSR_gflops, CSR_bandwith, CSR_pre_time;
    CSR_SpMV(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y_ref, &CSR_solve_time, &CSR_gflops, &CSR_bandwith, &CSR_pre_time);




    int maxBandwith = 0;
    double bandwith_average = 0;
    int num=0, maxNNZ = -1e9, minNNZ = 1e9;
    int* sssRowPtrA = new int[m+1];
    int* sssColIdxA = new int[nnzA/2];
    VALUE_TYPE* sssValA = new VALUE_TYPE[nnzA/2];
    VALUE_TYPE* sssDValA = new VALUE_TYPE[m];
    
    // convert CSR to SSS 
    for (int i = 0; i < m; i++)
    {
        sssDValA[i] = 0;
        sssRowPtrA[i] = num;
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
        {
            if (csrColIdxA[j] >= i)
            {
                if (csrColIdxA[j] == i)
                    sssDValA[i] = csrValA[j];
                break;
            }
            sssValA[num] = csrValA[j];
            sssColIdxA[num++] = csrColIdxA[j];
        }
        bandwith_average += abs(i - csrColIdxA[csrRowPtrA[i]]);
        maxBandwith = max(maxBandwith, abs(i - csrColIdxA[csrRowPtrA[i]]));
    }
    sssRowPtrA[m] = num;
    bandwith_average /= m;

    //calcu nnz feature
    double nnz_Std = 0.0;
    double nnz_average = nnzA/m;
    for (int i = 0; i < m; i++)
    {
        int nnzRow = csrRowPtrA[i+1]-csrRowPtrA[i];
        nnz_Std += (nnzRow-nnz_average)*(nnzRow-nnz_average);
        maxNNZ = max(maxNNZ, nnzRow);
        minNNZ = min(minNNZ, nnzRow);
    }
    nnz_Std /= m;
    nnz_Std = sqrt(nnz_Std);


    int *nRowPerRow = new int[m];
    int **Row2Row = new int*[m];
    long long conflicts = create_Conflict_Graph(sssRowPtrA, sssColIdxA, nRowPerRow, Row2Row, m);

    if (conflicts >= 2e9)
    {
        /* too many conflicts, we use naive SpMV directly */
        if (predict != 0)
        {
            printf("predict error!!!\n");
            ml_result << "error" << "," << CSR_gflops << endl;
        }
    }
    else
    {
        //calcu conflict feature
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

        
        double DC_Hybrid_predict_solve_time, DC_Hybrid_predict_gflops, DC_Hybrid_predict_bandwith, DC_Hybrid_predict_pre_time;
        if (predict != 0)
            DC_Hybrid_SymSpMV(m, nnzA, sssRowPtrA, sssColIdxA, sssValA, sssDValA, x, y, nRowPerRow, Row2Row, &DC_Hybrid_predict_solve_time, &DC_Hybrid_predict_gflops, &DC_Hybrid_predict_bandwith, &DC_Hybrid_predict_pre_time, threads, predict);
        else
            DC_Hybrid_predict_gflops = CSR_gflops;

        double DC_Hybrid_truth_solve_time, DC_Hybrid_truth_gflops, DC_Hybrid_truth_bandwith, DC_Hybrid_truth_pre_time;
        if (truth != 0)
            DC_Hybrid_SymSpMV(m, nnzA, sssRowPtrA, sssColIdxA, sssValA, sssDValA, x, y, nRowPerRow, Row2Row, &DC_Hybrid_truth_solve_time, &DC_Hybrid_truth_gflops, &DC_Hybrid_truth_bandwith, &DC_Hybrid_truth_pre_time, threads, truth);
        else
         DC_Hybrid_truth_gflops = CSR_gflops;
        
            
        ml_result << DC_Hybrid_predict_gflops << "," << DC_Hybrid_truth_gflops << endl;
    }
    



    ml_result.close();

    delete[] sssRowPtrA;
    delete[] sssColIdxA;
    delete[] sssValA;
    delete[] sssDValA;

    delete[] csrRowPtrA;
    delete[] csrColIdxA;
    delete[] csrValA;
}

