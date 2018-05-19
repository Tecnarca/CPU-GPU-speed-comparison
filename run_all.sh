#!/bin/bash

# $1 == smaller matrix size, $2 == biggest matrix size, $3 == step of the matrix size
# programs must be in the "bin" subdirectory

rm -rf csv/ #remove old csv files
mkdir csv/
echo "Running monocore program"
./bin/monocore $1 $2 $3
echo "Running multicore program"
./bin/multicore $1 $2 $3 4 #'4' is the number of threads, you might want change this
echo "Running OpenMP program"
./bin/openmp $1 $2 $3
echo "Running CUDA program"
./bin/cuda $1 $2 $3
echo "Running CUBLAS program"
./bin/cublas $1 $2 $3
echo "Done. Now run 'python graph_plot.py' to see the data"
