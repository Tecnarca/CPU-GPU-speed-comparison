# CPU-GPU-speed-comparison
A simple comparison between single thread program against multi-threading and CUDA, through Matrix multiplication and Matrix inversion
Project of Electronic calculators and computer networks course held by A. Mancini at UNIVPM 

The programs executes a Matrix multiplication and Matrix inversion storing the execution times of operations
On CUDA the programs stores also the transfer time of data from CPU and GPU and the return time
With a Python script, all execution and transfer time are plotted 
On the x-axis there are Matrix size, on the y-axis there are times

# Main Function
In this section some lines of code will be shown, the most important ones for the functioning of the programs, all written in C++ and Python


## Matrix Inversion
Matrix inversion is the critical section of 'Monocore.cpp','Multicore.cpp' and 'openmp.cpp', that is the same program of 'Multicore.cpp'
different only for the use of the directive `#pragma omp parallel` usueful for manage multithreading process

Matrix inversion with Gauss_Jordan formula: 
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

In 'cuda.cpp' Matrix inversion code is differt only for the reason that here Gauss reduction is made simultaneously on both the row and the column of a pivot:


```
 __global__ void gaussjordan(double *A,  double *I, int n, int i){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    double p;
    //this is still the same gauss jordan algorithm used in other files
    //only that it does not require transposing the matrix two times
    if(row<n && col<n) //to ensure we are withing the matrix boundaries
        if(row>i){ // limits operation to rows below the pivot point
            p = A[row*n+i]/A[i*n+i];
            I[row*n+col] -= I[i*n+col]*p;  // apply for every row member
            if(col>=i){ //limits to row members to the right of the pivot
                A[row*n+col] -= A[i*n+col]*p;  // apply only to members right of pivot
            }
        }
 }
```


A different approach has been used in the 'cublas.cu' programm where the Matrix Inversion has been made through LU factorization formula:
`cblas_sgemv(CblasColMajor,CblasNoTrans,dim,dim,alfa,A,dim,C,incx,beta,B,incy);` is the key part of the code needed for the LU factorization


```
//----------------------CUBLAS CHARGE CODE----------------------
        //copy the matrices A and B from RAM to GPU RAM         

        status = cudaMemcpy(gpu_A, A, data_size,cudaMemcpyHostToDevice); //copy gpu_A <-A

        //B = A*C on the CPU, the resulting vector is used later by the cuSolver
        cblas_sgemv(CblasColMajor,CblasNoTrans,dim,dim,alfa,A,dim,C,incx,beta,B,incy);

        status = cudaMemcpy(gpu_B, B, smaller_size,cudaMemcpyHostToDevice); //copy gpu_B <-B

        cusolverStatus = cusolverDnSgetrf_bufferSize(cuhandle,dim,dim,gpu_A,dim,&Lwork); //compute  buffer  size  and  prep.memory

        //----------------------CUBLAS PARALLEL CODE----------------------
        //getrf factorize the provided matrix to LU, getrs solve the generic system A*X=B, that computes the inverse because B = Identity
        //Reference: https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs

        cusolverStatus = cusolverDnSgetrf(cuhandle,dim,dim,gpu_A,dim,gpu_Work,gpu_pivot,gpu_info);
        cusolverStatus = cusolverDnSgetrs(cuhandle, CUBLAS_OP_N,dim,1,gpu_A,dim,gpu_pivot,gpu_B,dim,gpu_info);
```
 #multicore.cpp

 #cuda.cpp

If you are using the compiled version of the files, you can check what libraries you're missing by running the command
ldd bin/*
If you want to check a specific file (let's say, FILENAME), use
ldd bin/FILENAME
