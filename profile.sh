#!/bin/bash
#SBATCH -N 1
#SBATCH -n 96
#SBATCH --exclusive
#SBATCH --mem=360G
echo -e "matrix,CSR,MKL,MKLSYM,DCH" > performance_result.csv
echo -e "matrix,rows,nnz,nnz_max,nnz_std,BW,BW_avg,conf,conf_den,conf_max,conf_min,conf_std,Level" > samples.csv
cd matrix
for matrix in * 
do 
    echo -n "profiling "$matrix"..."
    cd ..
    numactl --cpunodebind=0 --membind=0 ./profiling ./matrix/$matrix/$matrix.mtx $matrix
    echo -e "\t\tdone..."
    cd matrix
done
