#include <cuda.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#define DEBUG 0
//If DEBUG is setted, the program will print the used matrices and the times on the stdout

using namespace std;

/*function marked with '__global__' are the GPU Kernels*/

//This reduces the matrix to upper triangular
 __global__ void upperReduction(float *A,  float *I, int n, int piv){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float p;
    //this is still the same gauss jordan algorithm used in other files
    if(i<n && j<n) //to ensure we are within the matrix boundaries
        if(i>piv){ // limits operation to rows below the pivot point
            p = A[i*n+piv]/A[piv*n+piv];
            I[i*n+j] -= I[piv*n+j]*p;  // apply for each row member
            if(j>=piv){ //limits to row members to the right of the pivot
                A[i*n+j] -= A[piv*n+j]*p;  // apply only to members right of pivot
            }
        }
 }

//Reduces the matrix to lower triangular matrix
  __global__ void lowerReduction(float *A,  float *I, int n, int piv){

    //same function of before, but row and col are reversed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float p;
    if(i<n && j<n)
        if(i<piv){
            p = A[i*n+piv]/A[piv*n+piv];
            I[i*n+j] -= I[piv*n+j]*p;
            if(j<=piv){
                A[i*n+j] -= A[piv*n+j]*p;
            }
        }
 }

//Scales down the matrix in respect of the elements on the diagonal
 __global__ void scale(float *A,  float *I, int h){
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row<h && col<h)//to ensure we are withing not the matrix boundaries
    {
        I[row*h+col]  /= A[row*(h+1)];
        A[row*h+col] /= A[row*(h+1)];
    }
}

__global__ void matrixMultiplication(float* A, float* B, float* C, int n) {

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float c=0;

    // each thread computes one element of the block sub-matrix (and therefore one non-overlapping sub-matrix of C)

    if (row<n && col<n) { //to ensure we are not withing the matrix boundaries
        for (int i = 0; i < n; i++) {
            c += A[row*n+i] * B[i*n+col];
        }

        C[row*n+col] = c;
    }
}


/* From utils.cpp */
extern float* createRandomMatrixArray(long, long, bool);
extern float* createIdentityMatrixArray(long);
extern float* createEmptyMatrixArray(long);
extern void print_array_as_matrix(float*, long, char*);
extern void saveTimeToFile(long, float, char*);
extern bool multipliedMatrixCudaIsCorrect(float*, float*, float*, long);


int main(int argc, char **argv){

    long min_dim, max_dim, step, dim, data_size; //Used to determine which matrix dimensions we will test
    float *A, *B, *C; //After moltiplication , C=A*B
    float *D, *M; //M=A, D=Identity and after inversion: D = A^-1, M=Identity
    float *gpu_A, *gpu_B, *gpu_C; //GPU Matrices
    float *gpu_inv_A, *gpu_inv_I;
    float time; //Will contain elapsed time returned by CUDA events, in milliseconds
    chrono::high_resolution_clock::time_point start, finish; //Used to implement time measurement
    chrono::duration<double> elapsed1, elapsed2; //Used to contain the elapsed time  
    cudaError_t status; //variable for error handling
    cudaEvent_t begin, stop; //used to the time measurement of the functions on the GPU
    cudaEventCreate(&begin); //initialize objects
    cudaEventCreate(&stop);

    // Print the usage command if too few parameters were passed
    if(argc != 4){
        cout << "Usage: " << argv[0] << " [min_dim] [max_dim] [step]" << endl;
        return -1;
    }

    min_dim = strtol(argv[1], NULL, 10);
    max_dim = strtol(argv[2], NULL, 10)+1; //'+1' means we will evaluate the "max_dim" value passed as a argument
    step = strtol(argv[3], NULL, 10);

    //for each 'dim' from 'min_dim' to 'max_dim', with the step we chosen 
    for(dim=min_dim;dim<max_dim;dim+=step){

        //Matrices are created and used as arrays
        A = createRandomMatrixArray(dim, dim, true); //true means "invertible"
        B = createRandomMatrixArray(dim, dim, false); //true means "not invertible"
        C = createEmptyMatrixArray(dim);

        //Number of bytes contained in one matrix
        data_size = dim*dim*sizeof(float);

        dim3 threadsPerBlock(dim, dim);
        dim3 blocksPerGrid(1, 1);
        if (dim*dim > 512){ //total amount of threads in a single block cannot exceed 1024 (with a maxwell nVidia GPU)
            threadsPerBlock.x = 512; 
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(float(dim)/float(threadsPerBlock.x));
            blocksPerGrid.y = ceil(float(dim)/float(threadsPerBlock.y));
        }

        //allocate memory to contain the matrices
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

        //BEGIN MATRICES MULTIPLICATION

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUDA CHARGE CODE----------------------
        //copy the matrices A and B from RAM to GPU RAM        

        status = cudaMemcpy(gpu_A, A, data_size, cudaMemcpyHostToDevice);
        status = cudaMemcpy(gpu_B, B, data_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize(); //to reassure the copy has ended

        //----------------------CUDA CHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure
        elapsed1 = finish - start; //compute time difference

        //elapsed.count() gives the time in seconds
        if(DEBUG) cout << "MUL_GCHR: With dimension " << dim << ", elapsed time: " << elapsed1.count() << " s" << endl;
        
        //Save how much time the load took
        saveTimeToFile(dim, elapsed1.count(), "csv/load_multiplication_CUDA.csv");

        cudaEventRecord(begin, 0); //begin "recording" operations on GPU
        
        //----------------------CUDA PARALLEL CODE----------------------
        //load and execute the kernel to multiplication into the GPU

        matrixMultiplication <<< blocksPerGrid, threadsPerBlock >>> (gpu_A, gpu_B, gpu_C, dim);

        //----------------------CUDA PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //GPU kernel calls are asynchronous, so this is necessary

        //Find how much time the GPU spent on computing. 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time, begin, stop);        

        if(DEBUG) cout << "MUL_PRLL: With dimension " << dim << ", elapsed time: " << time << " ms" << endl;

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUDA DISCHARGE CODE----------------------
        //Reading and paste back on RAM the result matrix        

        status = cudaMemcpy(C, gpu_C, data_size, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        //----------------------CUDA DISCHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure

        elapsed2 = finish - start; //compute time difference



       if(DEBUG) cout << "MUL_CCHR: With dimension " << dim << ", elapsed time: " << elapsed2.count() << " s" << endl;
       
       //Save how much time the read of the result took
       saveTimeToFile(dim, elapsed2.count(), "csv/read_multiplication_CUDA.csv");
       
       //Save how much time the whole computation took (load+calculations+read)
       //Note: 'time' is in milliseconds
       saveTimeToFile(dim, elapsed1.count()+elapsed2.count()+time/1000, "csv/multiplication_CUDA.csv");

        if(DEBUG){
            print_array_as_matrix(C,dim,"C ");
            bool correct = multipliedMatrixCudaIsCorrect(A,B,C,dim);
            if(!correct){
                cout << "Multiplied matrix is not correct, aborting..." << endl;
                return -1;
            }
        }


                
        //Free useless memory on the GPU and on the RAM
        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_C);
        free(B);
        free(C);

        //BEGIN MATRIX INVERSION

        D = createIdentityMatrixArray(dim);

        //M=A
        M = new float[dim*dim];
        for (int h = 0; h < dim; h++){
            for (int w = 0; w < dim; w++)
                    M[h*dim+w] = A[h*dim+w];
        }

        //Number of bytes contained in one matrix
        data_size = dim*dim*sizeof(float);

        //allocate memory to contain the matrices
        status = cudaMalloc((void**) &gpu_inv_A, data_size);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMalloc((void**) &gpu_inv_I, data_size);  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUDA CHARGE CODE----------------------
        
        status = cudaMemcpy(gpu_inv_A, M, data_size, cudaMemcpyHostToDevice);
        status = cudaMemcpy(gpu_inv_I, D, data_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        //----------------------CUDA CHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure
        elapsed1 = finish - start; //compute time difference
        if(DEBUG) cout << "INV_GCHR: With dimension " << dim << ", elapsed time: " << elapsed1.count() << " s" << endl;
        saveTimeToFile(dim, elapsed1.count(), "csv/load_inversion_CUDA.csv");

        cudaEventRecord(begin, 0);

        //----------------------CUDA PARALLEL CODE----------------------
        //the whole 'for' reduces the matrix to diagonal
        //each call computes from a different line of pivot (passed with 'i')
        //NOTE: every kernel call waits for the previous one to finish

        for(int i=0;i<dim-1;i++){ 
            upperReduction <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim, i);
        }
        for(int i=dim-1;i>0;i--){ 
            lowerReduction <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim, i);
        }

        //this function scales the starting A matrix to the identity, so "I" will be the correct inverse
        scale <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim); //reduce matrix to diagonal

        //----------------------CUDA PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time, begin, stop);

        if(DEBUG) cout << "INV_PRLL: With dimension " << dim << ", elapsed time: " << time << " ms" << endl;
      
        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUDA DISCHARGE CODE----------------------
        //Reads back M and D        

        status = cudaMemcpy(M, gpu_inv_A, data_size, cudaMemcpyDeviceToHost);
        status = cudaMemcpy(D, gpu_inv_I, data_size, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        //----------------------CUDA DISCHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure

        elapsed2 = finish - start; //compute time difference
         
        if(DEBUG) cout << "INV_CCHR: With dimension " << dim << ", elapsed time: " << elapsed2.count() << " s" << endl;
        saveTimeToFile(dim, elapsed2.count(), "csv/read_inversion_CUDA.csv");
        saveTimeToFile(dim, elapsed1.count()+elapsed2.count()+time/1000, "csv/inversion_CUDA.csv");

        if(DEBUG){
            print_array_as_matrix(D,dim,"D ");
            print_array_as_matrix(M,dim,"M ");
            bool correct = multipliedMatrixCudaIsCorrect(A,D,M,dim);
            if(!correct){
                cout << "Multiplied matrix is not correct, aborting..." << endl;
                return -1;
            }
        }

        //deallocate 
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
