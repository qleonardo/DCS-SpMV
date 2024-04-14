#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>

#include <sys/time.h>
#include "TDTree.h"
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

void CheckCorrectness(int m, VALUE_TYPE *y_ref, VALUE_TYPE *y)
{/**************************************CheckCorrectness******************************************/
   
    printf("Now Checking Correctness\n");
    
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (abs(y_ref[i] - y[i]) > 0.01 && abs(y_ref[i] - y[i]) > 0.01 * y_ref[i])
        {
            error_count++;
//             cout << "ROW [ " << i << " ], NNZ SPAN: "
//                     << "\t ref = " << y[i]
//                     << ", \t std = " << y_ref[i]
// //                 << ", \t error = " << y_ref[i] - y[i]
//                     << endl;
//            break;
        }

    if (error_count == 0)
        cout << "Check... PASS!" << endl;
    else
        cout << "Check... NO PASS! #Error = " << error_count << " out of " << m << " entries.\n" << endl;
}

#endif



#ifndef REORDERING
#define REORDERING

void sssRowReordering(int *RowRev, int *RowPerm, int *new_sssRowPtrA, int *new_sssColIdxA, VALUE_TYPE *new_sssDValA, VALUE_TYPE *new_sssValA, VALUE_TYPE *new_x, int *sssRowPtrA, 
                                                                                            int *sssColIdxA, VALUE_TYPE *sssDValA, VALUE_TYPE *sssValA, VALUE_TYPE *x, int m);

void csrRowReordering(int *RowRev, int *RowPerm, int *&csrRowPtrA, int *&csrColIdxA, VALUE_TYPE *&csrValA, int m, int nnzA);

#endif



#ifndef CONFLICT_GRAPH
#define CONFLICT_GRAPH

long long create_Conflict_Graph(int* sssRowPtrA, int* sssColIdxA, int *nRowPerRow, int **Row2Row, int m);

#endif



#ifndef CREATE_TREE
#define CREATE_TREE
int CreateTree(TDTree *TDTreeRoot, int **Row2Row, int *nRowPerRow, int *sssRowPtrA, int m);
#endif


#ifndef CMK
#define CMK
#include <queue>
void CMK_reordering(int m, int *csrRowPtrA, int *csrColIdxA, int *rev, int *perm);
#endif

#ifndef CSR_TO_SSS
#define CSR_TO_SSS

void convert_CSR_to_SSS(int m, int nnzA, int *sssRowPtrA, int *sssColIdxA, VALUE_TYPE *sssValA, VALUE_TYPE *sssDValA, int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *csrValA);
#endif