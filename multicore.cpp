#include <iostream>
#include <pthread.h>
#include <chrono>
#include <cstring>
#define DEBUG 1
#define MIN(a, b) (((a) > (b)) ? (b) : (a))

using namespace std;

void* thread_mat_inv(void*);
void* thread_mat_mul(void*);

extern int** createRandomMatrix(unsigned, unsigned, bool);
extern double** createIdentityMatrix(unsigned);
extern void print_matrix(int**, unsigned, char*);
extern void print_matrix(double**, unsigned, char*);
extern int** createEmpyMatrix(unsigned);


int **A, **B, **C; 
double **D;
unsigned dim;

struct coord{
	int x;
	int y;
	int p;
	bool sup;
};


int main(int argc, char **argv){

	long min_dim, max_dim, step; 
	chrono::high_resolution_clock::time_point start, finish;
	chrono::duration<double> elapsed;
	pthread_t* threads;
	int thread_number; // ToDo: controllare se il numero è migliorabile
	coord* params;

	//ToDo(?) si puo' mettere il path del file da salvare come argomento di input
	if(argc != 5){
		cout << "Usage: " << argv[0] << " [min_dim] [max_dim] [step] [thread_number]" << endl;
		return -1;
	}

	min_dim = strtol(argv[1], NULL, 10);
	max_dim = strtol(argv[2], NULL, 10)+1;
	step = strtol(argv[3], NULL, 10);
	thread_number = strtol(argv[4], NULL, 10);
	threads = new pthread_t[thread_number];
	params = new coord[thread_number];

	for(dim=min_dim;dim<max_dim;dim+=step){

		//ToDo: parallelizzare anche questa funzione (?)
		A = createRandomMatrix(dim, dim, true); // true means "invertible"
		B = createRandomMatrix(dim, dim, false); // true means "not invertible"
		C = createEmpyMatrix(dim);
		D = createIdentityMatrix(dim);

		if(DEBUG){
    		print_matrix(A,n,"A ");
  		}

  		if(DEBUG){
    		print_matrix(B,n,"B ");
  		}

		start = chrono::high_resolution_clock::now(); //start time measure


		//----------------------MULTITHREAD CODE----------------------

		for(int i=0; i<thread_number; i++){
			params[i].x = i*dim/thread_number; //what row we start multiplicating
			params[i].y = MIN((i+1)*dim/thread_number,dim); //what row we stop multiplicating
			pthread_create(&threads[i],NULL,thread_mat_mul,(void*)&params[i]);
		}
		for (int i=0; i<thread_number; i++) {
 		   pthread_join(threads[i],NULL);
		}

		//----------------------MULTITHREAD CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

		if(DEBUG){
    		print_matrix(C,dim,"MULT ");
  		}

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		cout << "MUL: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------MULTITHREAD CODE----------------------

		for(int i=0; i<thread_number; i++){
			params[i].x = i*dim/thread_number; //what row we start reducing
			params[i].y = MIN((i+1)*dim/thread_number,dim); //whatrow we stop reducing
			params[i].p = i;
			params[i].sup = true;
			pthread_create(&threads[i],NULL,thread_mat_inv,(void*)&params[i]);
		}

		for (int i=0; i<thread_number; i++) {
 		   pthread_join( threads[i],NULL);
		}
		
		if(DEBUG){
			print_matrix(A,dim,"A ");
    		print_matrix(D,dim,"D ");
  		}
		
		for(int i=0; i<thread_number; i++){
			params[i].sup = false;
			pthread_create(&threads[i],NULL,thread_mat_inv,(void*)&params[i]);
		}

		for (int i=0; i<thread_number; i++) {
 		   pthread_join( threads[i],NULL);
		}
		
		//----------------------MULTITHREAD CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure
		
		if(DEBUG){
    		print_matrix(D,dim,"DIAG ");
  		}

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi
				
		free(A);
		free(B);
		free(C);
		free(D);
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
	coord *v = (coord*)params;
	int p = params.p; //indice del pivot
	if(v->sup){
		//riduci a triangolare superiore
	} else {
		//riduci a diagonale e scala
	}
	pthread_exit((void*)0);
}