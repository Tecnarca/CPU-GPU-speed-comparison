#include <iostream>
#include <pthread.h>
#include <chrono>
#include <cstring>
#define DEBUG 0
#define MIN(a, b) (((a) > (b)) ? (b) : (a))
//If DEBUG is setted, will print the used matrices and the times on the stdout

using namespace std;

/* From utils.cpp */
extern float** createRandomMatrix(long, long, bool);
extern float** createIdentityMatrix(long);
extern void print_matrix(float**, long, char*);
extern float** createEmptyMatrix(long);
extern void saveTimeToFile(long, float, char*);
extern bool multipliedMatrixIsCorrect(float**, float**, float**, long);

/* For inverting and multiplicating matrices, these functions will be timed */
/* pThread requires them to have type and parameters be void* */
void* thread_mat_inv(void*);
void* thread_mat_mul(void*);

//Global variables because every thread will operate on them
float **A, **B, **C; //After multiplicating, C=A*B
float **D, **M; //M=A, D=Identity and after inversion: D = A^-1, M=Identity
long dim; //dim of the currently used matrices
int thread_number;

static pthread_barrier_t barrier;

//used to tell the thread from wich column or row he should start inverting/multiplicating
//and with wich step, passed parameters will be described in the thread creation line
struct coord{
	int x;
	int y;
};

int main(int argc, char **argv){

	long min_dim, max_dim, step; //Used to determine what matrix dimensions we will test
	chrono::high_resolution_clock::time_point start, finish; //Used to implement the timing
	chrono::duration<double> elapsed; //Will contain the elapsed time
	pthread_t* threads; //thread objects array
	coord* params; //every thread has is own parameters

	// Print the usage command if too few parameters were passed
	if(argc != 5){
		cout << "Usage: " << argv[0] << " [min_dim] [max_dim] [step] [thread_number]" << endl;
		return -1;
	}

	min_dim = strtol(argv[1], NULL, 10);
	max_dim = strtol(argv[2], NULL, 10)+1; //'+1' means we will evaluate the "max_dim" value passed as a argument
	step = strtol(argv[3], NULL, 10);

	//thread number is taken via input
	thread_number = strtol(argv[4], NULL, 10);
	threads = new pthread_t[thread_number];
	params = new coord[thread_number];

	//for every dim from min_dim to max_dim, with step 'step'
	for(dim=min_dim;dim<max_dim;dim+=step){

		//ToDo: parallelizzare anche questa funzione (?)
		A = createRandomMatrix(dim, dim, true); // true means "invertible"
		B = createRandomMatrix(dim, dim, false); // true means "not invertible"
		C = createEmptyMatrix(dim);
		D = createIdentityMatrix(dim);

		if(DEBUG){
    		print_matrix(A,dim,"A ");
    		print_matrix(B,dim,"B ");
  		}

  		//BEGIN MATRICES MULTIPLICATION

		start = chrono::high_resolution_clock::now(); //start time measure


		//----------------------MULTITHREAD CODE----------------------

		for(int i=0; i<thread_number; i++){
			//x and y are used to balance the load between threads.
			//in this case a thread computes the submatrix of the result.
			//the submatrix is from the column 'x' to the column 'y' 
			params[i].x = i*dim/thread_number; //what row we start multiplicating
			params[i].y = MIN((i+1)*dim/thread_number,dim); //what row we stop multiplicating (must be within the max 'dim' value)
			//Creation of the threads
			pthread_create(&threads[i],NULL,thread_mat_mul,(void*)&params[i]);
		}

		//waits for all threads to exit
		for (int i=0; i<thread_number; i++) {
 		   pthread_join(threads[i],NULL);
		}

		//----------------------MULTITHREAD CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

		if(DEBUG){
    		print_matrix(C,dim,"C ");
    		bool correct = multipliedMatrixIsCorrect(A,B,C,dim);
      		if(!correct){
       			cout << "Multiplied matrix is not correct, aborting..." << endl;
        		return -1;
      		}
  		}

		elapsed = finish - start; //compute time difference

		//elapsed.count() gives the time in seconds
		if(DEBUG) cout << "MUL: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		
		saveTimeToFile(dim, elapsed.count(), "csv/multiplication_MultiThread.csv");

		//M = A
      	M = new float*[dim];
      	for (int h = 0; h < dim; h++){
        	M[h] = new float[dim];
        	for (int w = 0; w < dim; w++)
                	M[h][w] = A[h][w];
      	}

      	//BEGIN MATRIX INVERSION

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------MULTITHREAD CODE----------------------

		pthread_barrier_init(&barrier, NULL, thread_number);

		for(int i=0; i<thread_number; i++){
			//x is used to balance the load between threads. y is unused.
			//to be balanced, every thread reduces the all the columns of index x*n (until x*n < dim), where n is integer
			params[i].x = i; //what columns we reduce
			//Creation of the threads
			pthread_create(&threads[i],NULL,thread_mat_inv,(void*)&params[i]);
		}

		//waits for all threads to exit
		for (int i=0; i<thread_number; i++) {
 		   pthread_join( threads[i],NULL);
		}

		//----------------------MULTITHREAD CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure
		
		if(DEBUG){
    		print_matrix(D,dim,"D ");
    		print_matrix(M,dim,"M ");
    		bool correct = multipliedMatrixIsCorrect(A,D,M,dim);
      		if(!correct){
        		cout << "Multiplied matrix is not correct, aborting..." << endl;
        		return -1;
      		}
  		}

		elapsed = finish - start; //compute time difference

		if(DEBUG) cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;

		saveTimeToFile(dim, elapsed.count(), "csv/inversion_MultiThread.csv");

		//Free because we will reallocate memory in the next for step
		free(A);
		free(B);
		free(C);
		free(D);
		free(M);
	}
	
	return 0;
}

void* thread_mat_mul(void* params) {
	coord *v = (coord*)params;
	int c;
	for(int i=v->x; i<v->y; i++){ //foreach row
		for(int j=0;j<dim;j++){ //foreach column
			c=0;
			for(int k=0; k<dim; k++) //foreach element
				c+= A[i][k]*B[k][j];
			C[i][j] = c;
		}
	}
	pthread_exit((void*)0);
}

void* thread_mat_inv(void* params) {
	coord *v = (coord*)params; //casting the parameters back to the right type
	double s;

		    //reduce M to upper triangular 
		    for(int piv=0; piv<dim; piv++){ //foreach row
		      //required sync accross all threads
			  //if not reassured, one thread will start overwriting a row currently used from another thread
			  pthread_barrier_wait(&barrier);
		      for(int i=piv+1+v->x; i<dim; i+=thread_number){ //foreach column
		        s = (double)M[i][piv]/M[piv][piv];
		        for(int j=0;j<dim;j++){ //for every element
		          D[i][j] -= (float)s*D[piv][j];
		          if(j>=piv) M[i][j] -= s*M[piv][j];
		        }
		      }
		    }

		    //reduce M to lower triangular
		    //the scaling operation is done within the first for, no need for another one
		    for(int piv=dim-1; piv>=0; piv--){ //foreach row
		      //required sync accross all threads
			  //if not reassured, one thread will start overwriting a row currently used from another thread
			  pthread_barrier_wait(&barrier);
		      for(int i=v->x; i<piv; i+=thread_number){ //foreach column
		        s = (double)M[i][piv]/M[piv][piv];
		        for(int j=0;j<dim;j++){ //for every element
		          D[i][j] -= (float)s*D[piv][j];
		          if(j<=piv) M[i][j] -= s*M[piv][j];
		        }
		      }
		    }

			//scales down the matrix
		    for(int i=v->x;i<dim;i+=thread_number){
		      s = M[i][i];
		      for(int j=0;j<dim;j++){
		        D[i][j]/=s;
		        M[i][j]/=s;
		      }
		    }


	pthread_exit((void*)0);
}
