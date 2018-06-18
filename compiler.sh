#!/bin/bash
# '-c' = compile and create the object file, without linking
# '-o' = output file
# '-std=c++11' = use the c++ 2011's standard
# '-w' = change the conversion warning from static string to char*
# '-lm' = link to the library "math"
# '-L/usr/local/cuda/lib64' = add /usr/local/cuda/lib64 to the paths from where you take the libraries
# '-lcudart' = link to the library CUDA
# '-lcublas' = link to the library cuBLAS
# '-lcusolver' = link to the cuBLAS's solver
# '-lopenblas' = link to cblas, it's used to pre-process the matrices before giving them to cuSolver
# nvcc it's the CUDA's compiler

mkdir -p objects/
mkdir -p bin/
nvcc -c src/cuda.cu -o objects/cuda.o -std=c++11 -w #create the object file for CUDA
nvcc -c src/cublas.cu -o objects/cublas.o -std=c++11 -w #create the object file for cublas
g++ -c src/monocore.cpp -o objects/monocore.o -std=c++11 -w #create the object file for the monocore
g++ -c src/multicore.cpp -o objects/multicore.o -std=c++11 -w #create the object file for the multicore
g++ -c src/utils.cpp -o objects/utils.o -std=c++11 #create the object file for utils
g++ -c src/openmp.cpp -o objects/openmp.o -std=c++11 -w -fopenmp #fopenmp says to the compiler to parallelize the "pragma parallel for"
g++ objects/monocore.o objects/utils.o -o bin/monocore -lm -std=c++11 #link monocore and utils to create the exe
g++ objects/multicore.o objects/utils.o -o bin/multicore -lm -lpthread -std=c++11 #linka multicore e utils to create the exe
g++ objects/cuda.o objects/utils.o -o bin/cuda -lm -L/usr/local/cuda/lib64 -lcudart #linka multicore e utils to create the exe (maybe some options are missing)
g++ objects/cublas.o objects/utils.o -o bin/cublas -lm -L/usr/local/cuda/lib64 -lcusolver -lcudart -lcublas -lopenblas #linka multicore e utils to create the exe(maybe some options are missing)
g++ objects/openmp.o objects/utils.o -o bin/openmp -lm -std=c++11 -fopenmp -lpthread #fopenmp says to che compiler to parallelize the "pragma parallel for"
