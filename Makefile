#compilers
CC=icpc

#GLOBAL_PARAMETERS
#VALUE_TYPE = double
#NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS

LIBS += -L . -lmetis -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L. -lTDTree -lmetis 

#backup
#$(CC) -xCORE-AVX2 -opt-prefetch=3 -Wno-deprecated-writable-strings -fopenmp -O3 main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
main:
	$(CC) -I /public1/soft/python/3.9.6-para/new-3.9.6/include/python3.9/ -std=c++11 -qopenmp -O3 main.cpp -Wl,-rpath . $(LIBS) -o csr.exe

color:
	$(CC) -xAVX -I /public1/soft/python/3.9.6-para/new-3.9.6/include/python3.9/ -std=c++11 -qopt-report=5 -qopenmp -O3 main.cpp -Wl,-rpath . $(LIBS) -o csr.exe -DCOLOR  
