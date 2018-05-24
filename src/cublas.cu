#include <cuda.h>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstring>
#include <cmath>
#define DEBUG 1
//If DEBUG is setted, will print the used matrices and the times on the stdout

using namespace std;

/* From utils.cpp */
extern void print_array_as_matrixT(float*, long, char*);
extern float* createRandomMatrixArray(long, long, bool);
extern float* createEmptyMatrixArray(long);
extern float* createIdentityMatrixArray(long);
extern void saveTimeToFile(long, float, char*);
extern bool multipliedMatrixCublasIsCorrect(float*, float*, float*, long);

int main(int argc, char **argv){

    long min_dim, max_dim, step, dim, data_size, smaller_size;
    float *A, *B, *C; //Cublas single precision requires matrices to be float* type, C=A*B when multiplicating and C=A^-1 when inverting
    float *gpu_A, *gpu_B, *gpu_C, *gpu_Work;//GPU Matrices
    int *gpu_pivot , *gpu_info , Lwork;   // pivots, info, worksp. size, used by cublas
    int info_gpu = 0;
    float time1,time2,time3; //Will contain elapsed time returned by CUDA events, in milliseconds
    float  alfa=1.0f; //costants for the cublas solver
    float  beta=0.f;
    int  incx=1, incy =1;
    cudaError_t status; //variable for error handling
    cudaEvent_t begin, stop; //used to time the functions on the GPU
    cublasStatus_t  stat; //CUBLAS functions status
    cublasHandle_t  handle; //CUBLAS context
    cusolverStatus_t  cusolverStatus;
    cusolverDnHandle_t  cuhandle;
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

    //for every dim from min_dim to max_dim, with step 'step'
    for(dim=min_dim;dim<max_dim;dim+=step){

        //Matrix as a sequential array, copied back from "S"
        A = createRandomMatrixArray(dim, dim, true); //true means "invertible"
        B = createRandomMatrixArray(dim, dim, false); //false means "invertible"
        C = createEmptyMatrixArray(dim);
 
        //Number of bytes contained in one matrix
        data_size = dim*dim*sizeof(*A);

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
            print_array_as_matrixT(A,dim,"A ");
            print_array_as_matrixT(B,dim,"B ");
        }

        //BEGIN MATRICES MULTIPLICATION

        stat = cublasCreate(&handle);

        cudaEventRecord(begin, 0); //start time measure

        //----------------------CUBLAS CHARGE CODE----------------------
        //copy the matrices A and B from RAM to GPU RAM 

        stat = cublasSetMatrix(dim,dim,sizeof(*A),A,dim,gpu_A,dim);//a -> gpu_A
        stat = cublasSetMatrix(dim,dim,sizeof(*B),B,dim,gpu_B,dim);//b -> gpu_B
        stat = cublasSetMatrix(dim,dim,sizeof(*C),C,dim,gpu_C,dim);//c -> gpu_C
        //----------------------CUBLAS CHARGE CODE----------------------
        cudaDeviceSynchronize(); //to reassure everything is in sync
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop); //end time measure
        cudaEventElapsedTime( &time1, begin, stop); //compute time difference

        if(DEBUG) cout << "MUL_GCHR: With dimension " << dim << ", elapsed time: " <<  time1 << " ms" << endl;

        //Save how much time the load took
        saveTimeToFile(dim, time1/1000, "csv/load_multiplication_CUBLAS.csv");     

        cudaEventRecord(begin, 0);
        //----------------------CUBLAS PARALLEL CODE----------------------
        //Sgemm is the only function provided to multiply matrices, the formula that follows is:
        // C := alfa*A*B + beta*C (uses C as "bias" matrix and puts the result in C itself)
        stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,dim,dim,dim,&alfa,gpu_A,dim,gpu_B,dim,&beta,gpu_C,dim);

        //----------------------CUBLAS PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        cudaEventElapsedTime( &time2, begin, stop);

        if(DEBUG) cout << "MUL_PRLL: With dimension " << dim << ", elapsed time: " << time2 << " ms" << endl;

        cudaEventRecord(begin, 0); //start time measure

        //----------------------CUBLAS DISCHARGE CODE----------------------
        //Reading and paste back on RAM the result matrix        

        stat=cublasGetMatrix(dim,dim,sizeof(*C),gpu_C,dim,C,dim); // gpu_C -> C
        
        cudaDeviceSynchronize();

        //----------------------CUBLAS DISCHARGE CODE----------------------

        cudaDeviceSynchronize(); //to reassure everything is in sync
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time3, begin, stop);
        

        if(DEBUG) cout << "MUL_CCHR: With dimension " << dim << ", elapsed time: " << time3 << " ms" << endl;
        
        //Save how much time the read of the result took
        saveTimeToFile(dim, time3/1000, "csv/read_multiplication_CUBLAS.csv");

        //Save how much time the whole computation took (load+calculations+read)
        saveTimeToFile(dim, (time1+time2+time3)/1000, "csv/multiplication_CUBLAS.csv");

        if(DEBUG){
            print_array_as_matrixT(C,dim,"C ");
            bool correct = multipliedMatrixCublasIsCorrect(A,B,C,dim);
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
        cublasDestroy(handle);   

        //BEGIN MATRIX INVERSION
        B = createIdentityMatrixArray(dim);
        C = createEmptyMatrixArray(dim);

        //creating cusolver handler
        cusolverStatus = cusolverDnCreate(&cuhandle); 

        //allocate memory to contain the matrices
        status = cudaMalloc((void**) &gpu_A, data_size);
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMalloc((void**) &gpu_B, dim*sizeof(float));  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMalloc((void**) &gpu_pivot, dim*sizeof(int));  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        status = cudaMalloc((void**) &gpu_info, dim*sizeof(int));  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        smaller_size = dim*sizeof(float);

        cudaEventRecord(begin, 0); //start time measure

        //----------------------CUBLAS CHARGE CODE----------------------
        //copy the matrices A and B from RAM to GPU RAM         

        status = cudaMemcpy(gpu_A, A, data_size,cudaMemcpyHostToDevice); //copy gpu_A <-A

        //B = A*C on the CPU, the resulting vector is used later by the cuSolver
        cblas_sgemv(CblasColMajor,CblasNoTrans,dim,dim,alfa,A,dim,C,incx,beta,B,incy);

        status = cudaMemcpy(gpu_B, B, smaller_size,cudaMemcpyHostToDevice); //copy gpu_B <-B

        cusolverStatus = cusolverDnSgetrf_bufferSize(cuhandle,dim,dim,gpu_A,dim,&Lwork); //compute  buffer  size  and  prep.memory

        //----------------------CUBLAS CHARGE CODE----------------------

        cudaDeviceSynchronize(); //to reassure everything is in sync
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time1, begin, stop);

        if(DEBUG) cout << "INV_GCHR: With dimension " << dim << ", elapsed time: " << time1 << " ms" << endl;
        saveTimeToFile(dim, time1/1000, "csv/load_inversion_CUBLAS.csv");

        status = cudaMalloc((void**) &gpu_Work, Lwork*sizeof(float));  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        cudaEventRecord(begin, 0);

        //----------------------CUBLAS PARALLEL CODE----------------------
        //getrf factorize the provided matrix to LU, getrs solve the generic system A*X=B, that computes the inverse because B = Identity
        //Reference: https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs

        cusolverStatus = cusolverDnSgetrf(cuhandle,dim,dim,gpu_A,dim,gpu_Work,gpu_pivot,gpu_info);
        cusolverStatus = cusolverDnSgetrs(cuhandle, CUBLAS_OP_N,dim,1,gpu_A,dim,gpu_pivot,gpu_B,dim,gpu_info);

        //----------------------CUBLAS PARALLEL CODE----------------------

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time2, begin, stop);

        if(DEBUG) cout << "INV_PRLL: With dimension " << dim << ", elapsed time: " << time2 << " ms" << endl;
        
        cudaEventRecord(begin, 0); //start time measure

        //----------------------CUBLAS DISCHARGE CODE----------------------
        //Reading and paste back on RAM the result matrix        

        status = cudaMemcpy (&info_gpu , gpu_info , sizeof(int), cudaMemcpyDeviceToHost );
        if(DEBUG) cout << "after getrf+getrs: info_gpu = " << info_gpu << endl;
        status = cudaMemcpy(C, gpu_B , dim*sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        //----------------------CUBLAS DISCHARGE CODE----------------------

        cudaDeviceSynchronize(); //to reassure everything is in sync
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time3, begin, stop);
         
        if(DEBUG) cout << "INV_CCHR: With dimension " << dim << ", elapsed time: " << time3 << " ms" << endl;
        saveTimeToFile(dim, time3/1000, "csv/read_inversion_CUBLAS.csv"); 
        saveTimeToFile(dim, (time1+time2+time3)/1000, "csv/inversion_CUBLAS.csv");

        if(DEBUG){
            print_array_as_matrixT(C,dim,"C ");
        }

        //deallocate things

        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_pivot);
        cudaFree(gpu_info);
        cudaFree(gpu_Work);
        free(A);
        free(B);
        free(C);
        cusolverStatus = cusolverDnDestroy(cuhandle);
    }
 
    cudaEventDestroy(begin);
    cudaEventDestroy(stop);

    return 0;
}
