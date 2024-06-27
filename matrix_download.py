import os
import sys
import csv

if os.path.exists("./matrix") == False:
    os.system("mkdir ./matrix")

filename = "./matrix_list.csv"

total = sum(1 for line in open(filename))
print(total)

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    for i in range(0, total):
        cur_row = next(csv_reader)
        if os.path.exists("./matrix/" + cur_row[1] + "/" + cur_row[1] + ".mtx") == False:
            matrix_url = "http://sparse-files.engr.tamu.edu/MM/" + cur_row[0] + "/" + cur_row[1] + ".tar.gz"
            os.system("wget " + matrix_url)
            os.system("tar -zxvf " + cur_row[1] + ".tar.gz -C ./matrix/")
            os.system("rm -rf " + cur_row[1] + ".tar.gz")

