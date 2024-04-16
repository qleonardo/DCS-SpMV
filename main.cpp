#include <iostream>
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
#include <Python.h>
using namespace std;


int main(int argc, char ** argv)
{
    // report precision of floating-point
    cout << "------------------------------------------------------" << endl;
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = "32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = "64-bit Double Precision";
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    cout << "PRECISION = " << precision << endl;
    cout << "------------------------------------------------------" << endl;

    int m, n, nnzA, isSymmetricA;
    int* csrRowPtrA;
    int* csrColIdxA;
    VALUE_TYPE* csrValA;
    //ex: ./spmv webbase-1M.mtx
    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
//    cout <<"argi = "<<argi<<endl;
    cout << "--------------" << filename << "--------------" << endl;

    srand(time(NULL));
    
    read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);
    printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
    cout << flush;


    // int *CMK_Rev = new int[m];
    // int *CMK_Perm = new int[m];
    // CMK_reordering(m, csrRowPtrA, csrColIdxA, CMK_Rev, CMK_Perm);

    // csrRowReordering(CMK_Rev, CMK_Perm, csrRowPtrA, csrColIdxA, csrValA, m, nnzA);
    // cout << "reorder finish!" << endl << flush;


    VALUE_TYPE *x = new VALUE_TYPE[m];
    VALUE_TYPE *y = new VALUE_TYPE[m];
    VALUE_TYPE *y_ref = new VALUE_TYPE[m];
    for (int i = 0; i < m; i++) x[i] = rand() % 10 / 10.0;


    double CSR_solve_time, CSR_gflops, CSR_bandwith, CSR_pre_time;
    CSR_SpMV(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y_ref, &CSR_solve_time, &CSR_gflops, &CSR_bandwith, &CSR_pre_time);
    printf("The preprocessing time = %4.2f ms, solving time =  %4.2f ms, throught = %4.2f gflops, bandwidth = %4.2f GB/s.\n\n", CSR_pre_time, CSR_solve_time, CSR_gflops, CSR_bandwith);
    cout << flush;



    double MKL_solve_time, MKL_gflops, MKL_bandwith, MKL_pre_time;
    MKL_SpMV(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, &MKL_solve_time, &MKL_gflops, &MKL_bandwith, &MKL_pre_time);
    CheckCorrectness(m, y_ref, y);
    printf("The preprocessing time = %4.2f ms, solving time =  %4.2f ms, throught = %4.2f gflops, bandwidth = %4.2f GB/s.\n\n", MKL_pre_time, MKL_solve_time, MKL_gflops, MKL_bandwith);
    cout << flush;



    double MKLSYM_solve_time, MKLSYM_gflops, MKLSYM_bandwith, MKLSYM_pre_time;
    MKL_SymSpMV(m, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, &MKLSYM_solve_time, &MKLSYM_gflops, &MKLSYM_bandwith, &MKLSYM_pre_time);
    CheckCorrectness(m, y_ref, y);
    printf("The preprocessing time = %4.2f ms, solving time =  %4.2f ms, throught = %4.2f gflops, bandwidth = %4.2f GB/s.\n\n", MKLSYM_pre_time, MKLSYM_solve_time, MKLSYM_gflops, MKLSYM_bandwith);
    cout << flush;


    int maxBandwith = 0;
    double bandwith_average = 0;
    int num=0, maxNNZ = -1e9, minNNZ = 1e9;
    int* sssRowPtrA = new int[m+1];
    int* sssColIdxA = new int[nnzA/2];
    VALUE_TYPE* sssValA = new VALUE_TYPE[nnzA/2];
    VALUE_TYPE* sssDValA = new VALUE_TYPE[m];

    if (csrRowPtrA[m/2] < nnzA/2)
    {
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
    }
    else
    {
        for (int i = 0; i < m; i++)
        {
            sssDValA[i] = 0;
            sssRowPtrA[i] = num;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            {
                if (csrColIdxA[j] <= i)
                {
                    if (csrColIdxA[j] == i)
                        sssDValA[i] = csrValA[j];
                    continue;
                }
                sssValA[num] = csrValA[j];
                sssColIdxA[num++] = csrColIdxA[j];
            }
            bandwith_average += abs(i - csrColIdxA[csrRowPtrA[i]]);
            maxBandwith = max(maxBandwith, abs(i - csrColIdxA[csrRowPtrA[i]]));
        }        
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

    cout << m << " " << nnzA << " " << maxNNZ << " " << nnz_Std << " " << maxBandwith << " " << bandwith_average << " " << flush;

    int ansLevel = -1;
    double ans = 1e9;

	cout << Level << endl << flush;
	double DC_Hybrid_solve_time, DC_Hybrid_gflops, DC_Hybrid_bandwith, DC_Hybrid_pre_time;
	DC_Hybrid_SymSpMV(m, nnzA, sssRowPtrA, sssColIdxA, sssValA, sssDValA, x, y, &DC_Hybrid_solve_time, &DC_Hybrid_gflops, &DC_Hybrid_bandwith, &DC_Hybrid_pre_time, 24, 4);
	CheckCorrectness(m, y_ref, y);
	printf("The preprocessing time = %4.2f ms, solving time =  %4.2f ms, throught = %4.2f gflops, bandwidth = %4.2f GB/s.\n", DC_Hybrid_pre_time, DC_Hybrid_solve_time, DC_Hybrid_gflops, DC_Hybrid_bandwith);
	
	double DC_solve_time, DC_gflops, DC_bandwith, DC_pre_time;
	DC_SymSpMV(m, nnzA, sssRowPtrA, sssColIdxA, sssValA, sssDValA, x, y, &DC_solve_time, &DC_gflops, &DC_bandwith, &DC_pre_time, 24, 4);
	CheckCorrectness(m, y_ref, y);
	printf("The preprocessing time = %4.2f ms, solving time =  %4.2f ms, throught = %4.2f gflops, bandwidth = %4.2f GB/s.\n", DC_pre_time, DC_solve_time, DC_gflops, DC_bandwith);

        
}

