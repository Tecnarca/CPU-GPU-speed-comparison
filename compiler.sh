#!/bin/bash
# '-c' = compila e crea il file oggetto, senza linkare
# '-o' = file di output
# '-std=c++11' = usa lo standard c++ 2011
# '-w' = sopprime gli warning di conversione da static string a char*
# '-lm' = linka la libreria "math"
# '-L/usr/local/cuda/lib64' = aggiunge /usr/local/cuda/lib64 ai percorsi da cui pescare le librerie
# '-lcudart' = linka la libreria CUDA
# '-lcublas' = linka la libreria cuBLAS
# '-lcusolver' = linka il solver di cuBLAS
# '-lopenblas' = linka cblas, serve per pre-processare le matrici prima di dare a cuSolver
# nvcc è il compilatore CUDA

mkdir -p objects/
mkdir -p bin/
nvcc -c src/cuda.cu -o objects/cuda.o -std=c++11 -w #crea il file oggetto per cuda
nvcc -c src/cublas.cu -o objects/cublas.o -std=c++11 -w #crea il file oggetto per cublas
g++ -c src/monocore.cpp -o objects/monocore.o -std=c++11 -w #crea il file oggetto per il monocore
g++ -c src/multicore.cpp -o objects/multicore.o -std=c++11 -w #crea il file oggetto per il multicore
g++ -c src/utils.cpp -o objects/utils.o -std=c++11 #crea il file oggetto per utils
g++ -c src/openmp.cpp -o objects/openmp.o -std=c++11 -w -fopenmp #fopenmp dice al compilatore di parallelizzare i "pragma parallel for"
g++ objects/monocore.o objects/utils.o -o bin/monocore -lm -std=c++11 #linka monocore e utils per creare l'exe
g++ objects/multicore.o objects/utils.o -o bin/multicore -lm -lpthread -std=c++11 #linka multicore e utils per creare l'exe
g++ objects/cuda.o objects/utils.o -o bin/cuda -lm -L/usr/local/cuda/lib64 -lcudart #linka multicore e utils per creare l'exe (probabilmente mancano delle opzioni)
g++ objects/cublas.o objects/utils.o -o bin/cublas -lm -L/usr/local/cuda/lib64 -lcusolver -lcudart -lcublas -lopenblas #linka multicore e utils per creare l'exe (probabilmente mancano delle opzioni)
g++ objects/openmp.o objects/utils.o -o bin/openmp -lm -std=c++11 -fopenmp -lpthread #fopenmp dice al compilatore di parallelizzare i "pragma parallel for"
