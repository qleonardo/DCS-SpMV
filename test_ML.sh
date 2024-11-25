#!/bin/bash
#SBATCH -N 1
#SBATCH -n 96
#SBATCH --exclusive
#SBATCH --mem=360G

echo -e "matrix,DCH(predict),DCH(truth)" > ml_result.csv
while read -r var1 var2 var3; do
    numactl --cpunodebind=0 --membind=0 ./test_ML ../../../matrix/$var1/$var1.mtx $var1 $var2 $var3
done < predict.txt
