#include <cuda.h>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#define DEBUG 0

using namespace std;

//ToDo: includere utils tramite utils.h
extern void print_array_as_matrix(int*, unsigned, char*);
extern void print_array_as_matrix(float*, unsigned, char*);
extern int* createRandomMatrixArray(unsigned, unsigned, bool);

int main(int argc, char **argv){

    long min_dim, max_dim, step, dim, data_size, smaller_size;
    int *S;
    float *A, *B, *C;
    float *gpu_A, *gpu_B, *gpu_C, *gpu_Work;
    int *gpu_pivot , *gpu_info , Lwork;   // pivots , info , worksp. size
    int info_gpu = 0;
    float time;
    float  alfa=1.0f;
    float  beta=0.0f;
    int  incx=1, incy =1;
    chrono::high_resolution_clock::time_point start, finish;
    chrono::duration<double> elapsed; 
    cudaError_t status;
    cudaEvent_t begin, stop;
    cublasStatus_t  stat; //CUBLAS functions status
    cublasHandle_t  handle; //CUBLAS context
    cusolverStatus_t  cusolverStatus;
    cusolverDnHandle_t  cuhandle;
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

        S = createRandomMatrixArray(dim, dim, true); //true means "invertible"
        A = new float[dim*dim];
        for(int i=0;i<dim;i++) for(int j=0;j<dim;j++) A[i*dim+j] = (float)S[i*dim+j];
        free(S);
        S = createRandomMatrixArray(dim, dim, false); //false means "invertible"
        B = new float[dim*dim];
        for(int i=0;i<dim;i++) for(int j=0;j<dim;j++) B[i*dim+j] = (float)S[i*dim+j];
        free(S);
        C = new float[dim*dim];
        for(int i=0;i<dim;i++) for(int j=0;j<dim;j++) C[i*dim+j] = 0;

        data_size = dim*dim*sizeof(float);

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

        stat = cublasCreate(&handle);

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUBLAS CHARGE CODE----------------------

        stat = cublasSetMatrix(dim,dim,data_size,A,dim,gpu_A,dim);//a -> gpu_A
        stat = cublasSetMatrix(dim,dim,data_size,B,dim,gpu_B,dim);//b -> gpu_B
        stat = cublasSetMatrix(dim,dim,data_size,C,dim,gpu_C,dim);//c -> gpu_C

        cudaDeviceSynchronize();

        //----------------------CUBLAS CHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure
        elapsed = finish - start; //compute time difference

        cout << "MUL_GCHR: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
        
        cudaEventRecord(begin, 0);
        
        //----------------------CUBLAS PARALLEL CODE----------------------

        // C := alfa*A*B + beta*C;
        stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,dim,dim,dim,&alfa,gpu_A,dim,gpu_B,dim,&beta,gpu_C,dim);

        //----------------------CUBLAS PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time, begin, stop);        

        cout << "MUL_PRLL: With dimension " << dim << ", elapsed time: " << time << " ms" << endl;

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUBLAS DISCHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        stat=cublasGetMatrix(dim,dim,data_size,gpu_C,dim,C,dim); // gpu_C -> C

        cudaDeviceSynchronize();

        //----------------------CUBLAS DISCHARGE CODE----------------------

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
        cublasDestroy(handle);   

        B = new float[dim];
        C = new float[dim];

        for(int i=0;i<dim;i++) B[i] = 0.0;                //  initialize B
        for(int i=0;i<dim;i++) C[i] = 1.0;    // C - N-vector  of ones

        cusolverStatus = cusolverDnCreate (& cuhandle ); 

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

        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUBLAS CHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        status = cudaMemcpy(gpu_A, A, data_size,cudaMemcpyHostToDevice);      // copy d_A <-A

        //moltiplica B = A*C
        cblas_sgemv(CblasColMajor,CblasNoTrans,dim,dim,alfa,A,dim,C,incx,beta,B,incy);

        status = cudaMemcpy(gpu_B, B, smaller_size,cudaMemcpyHostToDevice);      // copy d_B <-B

        cusolverStatus = cusolverDnSgetrf_bufferSize(cuhandle,dim,dim,gpu_A,dim,&Lwork);      //  compute  buffer  size  and  prep.memory

        cudaDeviceSynchronize();

        //----------------------CUBLAS CHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure
        elapsed = finish - start; //compute time difference
        cout << "INV_GCHR: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
        

        status = cudaMalloc((void**) &gpu_Work, Lwork*sizeof(float));  
        
        if(status!=cudaSuccess){
            cout << cudaGetErrorString(status) << " in " << __FILE__ << " at line " << __LINE__ << endl;
        }

        cudaEventRecord(begin, 0);

        //----------------------CUBLAS PARALLEL CODE----------------------

        cusolverStatus = cusolverDnSgetrf(cuhandle,dim,dim,gpu_A,dim,gpu_Work,gpu_pivot,gpu_info);
        cusolverStatus = cusolverDnSgetrs(cuhandle, CUBLAS_OP_N,dim,1,gpu_A,dim,gpu_pivot,gpu_B,dim,gpu_info);

        //----------------------CUBLAS PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time, begin, stop);

        cout << "INV_PRLL: With dimension " << dim << ", elapsed time: " << time << " ms" << endl;
      
        start = chrono::high_resolution_clock::now(); //start time measure

        //----------------------CUBLAS DISCHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        status = cudaMemcpy (&info_gpu , gpu_info , sizeof(int), cudaMemcpyDeviceToHost );
        if(DEBUG) cout << "after getrf+getrs: info_gpu = " << info_gpu << endl;
        status = cudaMemcpy(C, gpu_B , dim*sizeof(float), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        //----------------------CUBLAS DISCHARGE CODE----------------------

        finish = chrono::high_resolution_clock::now(); //end time measure

        elapsed = finish - start; //compute time difference
         
        cout << "INV_CCHR: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
         
        if(DEBUG){
            print_array_as_matrix(C,dim,"C ");
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
        status = cudaDeviceReset();
    }
 
    cudaEventDestroy(begin);
    cudaEventDestroy(stop);

    return 0;
}