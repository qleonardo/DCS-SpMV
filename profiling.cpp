#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "read_mtx.h"
#include "MKL_SpMV.h"
#include "MKL_SymSpMV.h"
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

    char  *filename, *matrix_name;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
        matrix_name = argv[argi];
    }

    // srand(time(NULL));
    read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);

    ofstream performance_result;
    performance_result.open("performance_result.csv", ios::app);
    performance_result << matrix_name << ",";

    ofstream samples;
    samples.open("samples.csv", ios::app);
    samples << matrix_name << ",";

    VALUE_TYPE *x = new VALUE_TYPE[m];
    VALUE_TYPE *y = new VALUE_TYPE[m];
    VALUE_TYPE *y_ref = new VALUE_TYPE[m];
    for (int i = 0; i < m; i++) 
        x[i] = (rand() % 100 - 100) / 100.0;




    double CSR_solve_time, CSR_gflops, CSR_bandwith, CSR_pre_time;
    CSR_SpMV(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y_ref, &CSR_solve_time, &CSR_gflops, &CSR_bandwith, &CSR_pre_time);
    performance_result << CSR_gflops << ",";




    double MKL_solve_time, MKL_gflops, MKL_bandwith, MKL_pre_time;
    MKL_SpMV(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, &MKL_solve_time, &MKL_gflops, &MKL_bandwith, &MKL_pre_time);
    CheckCorrectness(m, y_ref, y);
    performance_result << MKL_gflops << ",";




    double MKLSYM_solve_time, MKLSYM_gflops, MKLSYM_bandwith, MKLSYM_pre_time;
    MKL_SymSpMV(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, &MKLSYM_solve_time, &MKLSYM_gflops, &MKLSYM_bandwith, &MKLSYM_pre_time);
    CheckCorrectness(m, y_ref, y);
    performance_result << MKLSYM_gflops << ",";




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

    samples << m << "," << nnzA << "," << maxNNZ << "," << nnz_Std << "," << maxBandwith << "," << bandwith_average << ",";

    int *nRowPerRow = new int[m];
    int **Row2Row = new int*[m];
    long long conflicts = create_Conflict_Graph(sssRowPtrA, sssColIdxA, nRowPerRow, Row2Row, m);

    if (conflicts >= 2e9)
    {
        /* too many conflicts, we use naive SpMV directly */
        performance_result << CSR_gflops << endl;
        samples << conflicts << "," << 2500000000 << "," << 2500000000 << "," << 2500000000 << "," << 0 << endl << flush;
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
        samples << conflict_average << "," << maxConflict << "," << minConflict << "," << conflict_Std << "," << flush;

        // profiling
        int ansLevel = 0;
        double ansGflops = CSR_gflops;
        for (int Level = 1; Level <= 6; Level++)
        {
            double DC_Hybrid_solve_time, DC_Hybrid_gflops, DC_Hybrid_bandwith, DC_Hybrid_pre_time;
            DC_Hybrid_SymSpMV(m, nnzA, sssRowPtrA, sssColIdxA, sssValA, sssDValA, x, y, nRowPerRow, Row2Row, &DC_Hybrid_solve_time, &DC_Hybrid_gflops, &DC_Hybrid_bandwith, &DC_Hybrid_pre_time, threads, Level);
            if (!CheckCorrectness(m, y_ref, y)) continue;
        
            if (DC_Hybrid_gflops > ansGflops + 0.2)
            {
                ansGflops = DC_Hybrid_gflops;
                ansLevel = Level;
            }
        }

        samples << ansLevel << endl;
        performance_result << ansGflops << endl;
    }
    



    performance_result.close();
    samples.close();

    delete[] sssRowPtrA;
    delete[] sssColIdxA;
    delete[] sssValA;
    delete[] sssDValA;

    delete[] csrRowPtrA;
    delete[] csrColIdxA;
    delete[] csrValA;

    delete[] x;
    delete[] y;
    delete[] y_ref;

    for(int i = 0; i < m; i++)
        delete Row2Row[i];
    delete[] Row2Row;
    delete[] nRowPerRow;
}

