#include <cuda.h>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstring>
#include <cmath>
#define DEBUG 0

using namespace std;

//ToDo: includere utils tramite utils.h
extern void print_array_as_matrix(int*, long, char*);
extern void print_array_as_matrix(float*, long, char*);
extern int* createRandomMatrixArray(long, long, bool);
extern void saveTimeToFile(long, double, char*);

int main(int argc, char **argv){

    long min_dim, max_dim, step, dim, data_size, smaller_size;
    int *S;
    float *A, *B, *C;
    float *gpu_A, *gpu_B, *gpu_C, *gpu_Work;
    int *gpu_pivot , *gpu_info , Lwork;   // pivots , info , worksp. size
    int info_gpu = 0;
    float time1,time2,time3;
    float  alfa=1.0f;
    float  beta=0.0f;
    int  incx=1, incy =1;
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

        cudaEventRecord(begin, 0); //start time measure

        //----------------------CUBLAS CHARGE CODE----------------------

        stat = cublasSetMatrix(dim,dim,data_size,A,dim,gpu_A,dim);//a -> gpu_A
        stat = cublasSetMatrix(dim,dim,data_size,B,dim,gpu_B,dim);//b -> gpu_B
        stat = cublasSetMatrix(dim,dim,data_size,C,dim,gpu_C,dim);//c -> gpu_C

        //----------------------CUBLAS CHARGE CODE----------------------
        cudaDeviceSynchronize(); //to reassure everything is in sync
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time1, begin, stop);

        if(DEBUG) cout << "MUL_GCHR: With dimension " << dim << ", elapsed time: " <<  time1 << " ms" << endl;
        saveTimeToFile(dim, time1/1000, "csv/load_multiplication_CUBLAS.csv");     

        cudaEventRecord(begin, 0);
        //----------------------CUBLAS PARALLEL CODE----------------------

        // C := alfa*A*B + beta*C;
        stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,dim,dim,dim,&alfa,gpu_A,dim,gpu_B,dim,&beta,gpu_C,dim);

        //----------------------CUBLAS PARALLEL CODE---------------------- 

        cudaDeviceSynchronize(); //to reassure everything is in sync

        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);

        cudaEventElapsedTime( &time2, begin, stop);

        if(DEBUG) cout << "MUL_PRLL: With dimension " << dim << ", elapsed time: " << time2 << " ms" << endl;

        cudaEventRecord(begin, 0); //start time measure

        //----------------------CUBLAS DISCHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        stat=cublasGetMatrix(dim,dim,data_size,gpu_C,dim,C,dim); // gpu_C -> C

        cudaDeviceSynchronize();

        //----------------------CUBLAS DISCHARGE CODE----------------------

        cudaDeviceSynchronize(); //to reassure everything is in sync
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime( &time3, begin, stop);
        

        if(DEBUG) cout << "MUL_CCHR: With dimension " << dim << ", elapsed time: " << time3 << " ms" << endl;
        saveTimeToFile(dim, time3/1000, "csv/read_multiplication_CUBLAS.csv");

        saveTimeToFile(dim, (time1+time2+time3)/1000, "csv/multiplication_CUBLAS.csv");

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

        cudaEventRecord(begin, 0); //start time measure

        //----------------------CUBLAS CHARGE CODE----------------------
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

        status = cudaMemcpy(gpu_A, A, data_size,cudaMemcpyHostToDevice);      // copy d_A <-A

        //moltiplica B = A*C
        cblas_sgemv(CblasColMajor,CblasNoTrans,dim,dim,alfa,A,dim,C,incx,beta,B,incy);

        status = cudaMemcpy(gpu_B, B, smaller_size,cudaMemcpyHostToDevice);      // copy d_B <-B

        cusolverStatus = cusolverDnSgetrf_bufferSize(cuhandle,dim,dim,gpu_A,dim,&Lwork);      //  compute  buffer  size  and  prep.memory

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
        //si basa sul fatto che i caricamenti sono sincroni, mentre l'esecuzione parallela no        

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
    }
 
    cudaEventDestroy(begin);
    cudaEventDestroy(stop);

    return 0;
}
