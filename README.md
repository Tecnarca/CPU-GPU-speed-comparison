# CPU-GPU-speed-comparison
A simple comparison between single thread program against multi-threading and CUDA, through Matrix multiplication and Matrix inversion.
Project of Electronic calculators and computer networks course held by A. Mancini at UNIVPM 

The programs executes a Matrix multiplication and Matrix inversion storing the execution times of operations.
On CUDA the programs stores also the transfer time of data from CPU and GPU and the return time.
With a Python script, all execution and transfer time are plotted. 
On the x-axis there are Matrix size, on the y-axis there are times.

# Programs
In this section some lines of code will be shown, the most important ones for the functioning of the programs, all written in C++ and Python.


## monocore.cpp

The critical sections of 'Monocore' program are Matrix inversion, made through Gauss-Jordan formula: 
```
void mat_inv(double **M, double **D, long dim){
  double p;
  for(int z=0; z<2; z++){       //done two times:
                                //reduce M to upper triangular 
    for(int k=0; k<dim; k++){   //foreach row
      p = M[k][k];
      for(int j=k; j<dim; j++){ //foreach column
        M[k][j] = M[k][j]/p;
        D[k][j] = D[k][j]/p;
        for(int i=k+1;i<dim;i++){ //for every element
          M[i][j] -= M[i][k]*M[k][j];
          D[i][j] -= M[i][k]*D[k][j];
        }
      }
    }
```

 #multicore.cpp

 #cuda.cpp

If you are using the compiled version of the files, you can check what libraries you're missing by running the command
ldd bin/*
If you want to check a specific file (let's say, FILENAME), use
ldd bin/FILENAME
