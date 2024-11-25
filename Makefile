#compilers
CC=icpx

LIBS += -L . -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -L. -lDC 

all: Profile ML

Profile: Profile
	$(CC) -I -std=c++11 -qopenmp -O3 profiling.cpp -Wl,-rpath . $(LIBS) -o profiling

ML: ML
	$(CC) -I -std=c++11 -qopenmp -O3 test_ML.cpp -Wl,-rpath . $(LIBS) -o test_ML

clean:
	rm -f profiling
	rm -f test_ML
	rm -f predict.txt
	rm -f performance_result.csv
	rm -f samples.csv
	rm -f ml_result.csv
