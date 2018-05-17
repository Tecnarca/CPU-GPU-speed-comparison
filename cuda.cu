#include <cuda.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#define DEBUG 0

using namespace std;

 __global__ void gaussjordan(double *A,  double *I, int n, int i){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    double p;

    if(row<n && col<n)
        if(row>i){ // limits operation to rows below the pivot point
            p = A[row*n+i]/A[i*n+i];
            I[row*n+col] -= I[i*n+col]*p;  // apply for every row member
            if(col>=i){ //limits to row members to the right of the pivot
                A[row*n+col] -= A[i*n+col]*p;  // apply only to members right of pivot
            }
        }
 }


 __global__ void scale(double *A,  double *I, int h){
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row<h && col<h)
        if(A[row*(h+1)]!=0){
            // scale down to identity in each cell
            I[row*h+col]  /= A[row*(h+1)];
            A[row*h+col] /= A[row*(h+1)]; 
        }
}

__global__ void matrixMultiplication(int* A, int* B, int* C, int n) {

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    double c=0;

    // each thread computes one element of the block sub-matrix

    if (row<n && col<n) {
        for (int i = 0; i < n; i++) {
            c += A[row*n+i] * B[i*n+col];
        }

        C[row*n+col] = c;
    }
}


//ToDo: includere utils tramite utils.h
extern int* createRandomMatrixArray(unsigned, unsigned, bool);
extern double* createIdentityMatrixArray(unsigned);
extern int* createEmpyMatrixArray(unsigned);
extern void print_array_as_matrix(int*, unsigned, char*);
extern void print_array_as_matrix(double*, unsigned, char*);


int main(int argc, char **argv){

    long min_dim, max_dim, step, dim, data_size;
    int *A, *B, *C; //dopo moltiplicazione, C = A*B
    double *D, *M; //dopo inversione, M = I && D = A^-1
    int *gpu_A, *gpu_B, *gpu_C;
    double *gpu_inv_A, *gpu_inv_I;
    float time;
    chrono::high_resolution_clock::time_point start, finish;
    chrono::duration<double> elapsed; 
    cudaError_t status;
    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    if(argc != 4){
        cout << "Usage: " << argv[0] << " [min_dim] [max_dim] [step]" << endl;
        return -1;
    }

    min_dim = strtol(argv[1], NULL, 10);
    max_dim = strtol(argv[2], NULL, 10)+1;
    step = strtol(argv[3], NULL, 10);

    for(dim=min_dim;dim<max_dim;dim+=step){

        //Matrix as a sequential array
        A = createRandomMatrixArray(dim, dim, true); //true means "invertible"
        B = createRandomMatrixArray(dim, dim, false); //true means "not invertible"
        C = createEmpyMatrixArray(dim);

        data_size = dim*dim*sizeof(int);

        dim3 threadsPerBlock(dim, dim);
        dim3 blocksPerGrid(1, 1);
        if (dim*dim > 512){ //total amount of threads in a single block cannot exceed 1024
            threadsPerBlock.x = 512; 
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(dim)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(dim)/double(threadsPerBlock.y));
        }

        status = cudaMalloc((void**) &gpu_A, data_size);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMalloc((void**) &gpu_B, data_size);  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMalloc((void**) &gpu_C, data_size);  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        if(DEBUG){
            print_array_as_matrix(A,dim,"A ");
            print_array_as_matrix(B,dim,"B ");
        }

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUDA CHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        status = cudaMemcpy(gpu_A, A, data_size, cudaMemcpyHostToDevice);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMemcpy(gpu_B, B, data_size, cudaMemcpyHostToDevice);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        cudaDeviceSynchronize();

        //----------------------CUDA CHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure
        elapsed = finish - start; //compute time difference

        cout << "MUL_GCHR: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
        
        cudaEventRecord(begin, 0);
        
        //----------------------CUDA PARALLEL CODE----------------------

        matrixMultiplication <<< blocksPerGrid, threadsPerBlock >>> (gpu_A, gpu_B, gpu_C, dim);

        //----------------------CUDA PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time, begin, stop);        

        cout << "MUL_PRLL: With dimension " << dim << ", elapsed time: " << time << " ms" << endl;

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUDA DISCHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        status = cudaMemcpy(C, gpu_C, data_size, cudaMemcpyDeviceToHost);

        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        cudaDeviceSynchronize();

        //----------------------CUDA DISCHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure

        elapsed = finish - start; //compute time difference

       cout << "MUL_CCHR: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;

        if(DEBUG){
            print_array_as_matrix(C,dim,"MULT ");
        }


                

        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_C);
        free(B);
        free(C);

        D = createIdentityMatrixArray(dim);

        M = new double[dim*dim];

        for (int h = 0; h < dim; h++){
            for (int w = 0; w < dim; w++)
                    M[h*dim+w] = A[h*dim+w];
        }

        data_size = dim*dim*sizeof(double);

        status = cudaMalloc((void**) &gpu_inv_A, data_size);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMalloc((void**) &gpu_inv_I, data_size);  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMemcpy(gpu_inv_I, D, data_size, cudaMemcpyHostToDevice);

        //----------------------CUDA CHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        status = cudaMemcpy(gpu_inv_A, M, data_size, cudaMemcpyHostToDevice);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        start = chrono::high_resolution_clock::now(); //start time measure

        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        cudaDeviceSynchronize();

        //----------------------CUDA CHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure
        elapsed = finish - start; //compute time difference
        cout << "INV_GCHR: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
        
        cudaEventRecord(begin, 0);

        //----------------------CUDA PARALLEL CODE----------------------

        for(int i=0;i<dim;i++){ //reduce matrix to diagonal, and everytime waits for completition of previous kernel
            gaussjordan <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim, i);
        }
    
        scale <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim); //reduce matrix to diagonal

        //----------------------CUDA PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time, begin, stop);

        cout << "INV_PRLL: With dimension " << dim << ", elapsed time: " << time << " ms" << endl;
      
        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUDA DISCHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        status = cudaMemcpy(M, gpu_inv_A, data_size, cudaMemcpyDeviceToHost);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMemcpy(D, gpu_inv_I, data_size, cudaMemcpyDeviceToHost);

        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        cudaDeviceSynchronize();

        //----------------------CUDA DISCHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure

        elapsed = finish - start; //compute time difference
         
        cout << "INV_CCHR: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
         
        if(DEBUG){
            print_array_as_matrix(D,dim,"D ");
            print_array_as_matrix(M,dim,"M ");
        }

        //deallocate things

        cudaFree(gpu_inv_A);
        cudaFree(gpu_inv_I);
        free(A);
        free(D);
        free(M);        

    }
 
    cudaEventDestroy(begin);
    cudaEventDestroy(stop);  

    return 0;
}