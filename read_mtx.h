#ifndef READ_MTX_H
#define READ_MTX_H

#include "mmio.h"
#include "common.h"

int read_mtx(char  * filename, int* m_add, int *n_add, int *nnzA_add, int **csrRowPtrA_add, int **csrColIdxA_add, VALUE_TYPE **csrValA_add)
{
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;
    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    
    //printf("00000\n");
    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
    
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;
    
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }
    
    if ( mm_is_complex( matcode ) )
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        return -3;
    }
    
    //printf("111111\n");
    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }
    //printf("222222\n");
    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    //printf("3333333,%d\n",ret_code);
    
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }
    
    
    
    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));
    
    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE *csrValA_tmp    = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));
    
    if(csrRowIdxA_tmp==NULL || csrColIdxA_tmp==NULL || csrValA_tmp==NULL)
        return -2;
    
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    
    //printf("222222\n");
    int i;
    for (i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;
        int returnvalue;
        
        if (isReal)
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }
        
        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }
    
    if (f != stdin)
        fclose(f);
    
    if (isSymmetric)
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }
    
    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;
    
    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }
    
    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));
    
    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
    
    if (isSymmetric)
    {
        for ( i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
                
                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }
    
    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);
    
    //printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
    *m_add=m;
    *n_add=n;
    *nnzA_add=nnzA;
    *csrColIdxA_add=csrColIdxA;
    *csrValA_add=csrValA;
    *csrRowPtrA_add=csrRowPtrA;
    
    
    return 0;
}
#endif