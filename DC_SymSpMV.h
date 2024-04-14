#include "common.h"
#include "TDTree.h"
#include "omp.h"

void TDTree_SymSpMV_kernel(char ** userArgs, TDTreeArgs *treeArgs);



double TDTree_SymSpMV(int            m,
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
                            int             Level);
