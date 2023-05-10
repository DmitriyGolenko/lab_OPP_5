build: mpicxx -mt_mpi main.cpp -std-c++11

run: mpiexec -n 16 ./out
