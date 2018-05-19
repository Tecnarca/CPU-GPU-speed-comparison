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

rm *.o
nvcc -c src/cuda.cu -o cuda.o -std=c++11 -w #crea il file oggetto per cuda
nvcc -c src/cublas.cu -o cublas.o -std=c++11 -w #crea il file oggetto per cublas
g++ -c src/monocore.cpp -o monocore.o -std=c++11 -w #crea il file oggetto per il monocore
g++ -c src/multicore.cpp -o multicore.o -std=c++11 -w #crea il file oggetto per il multicore
g++ -c src/utils.cpp -o utils.o -std=c++11 #crea il file oggetto per utils
g++ -c src/openmp.cpp -o openmp.o -std=c++11 -w -fopenmp #fopenmp dice al compilatore di parallelizzare i "pragma parallel for"
g++ monocore.o utils.o -o monocore -lm -std=c++11 #linka monocore e utils per creare l'exe
g++ multicore.o utils.o -o multicore -lm -lpthread -std=c++11 #linka multicore e utils per creare l'exe
g++ cuda.o utils.o -o cuda -lm -L/usr/local/cuda/lib64 -lcudart #linka multicore e utils per creare l'exe (probabilmente mancano delle opzioni)
g++ cublas.o utils.o -o cublas -lm -L/usr/local/cuda/lib64 -lcusolver -lcudart -lcublas -lopenblas #linka multicore e utils per creare l'exe (probabilmente mancano delle opzioni)
g++ openmp.o utils.o -o openmp -lm -std=c++11 -fopenmp -lpthread #fopenmp dice al compilatore di parallelizzare i "pragma parallel for"
rm *.o