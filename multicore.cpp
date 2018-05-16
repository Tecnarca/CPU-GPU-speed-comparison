#include <iostream>
#include <pthread.h>
#include <chrono>
#include <cstring>
#define DEBUG 0
#define MIN(a, b) (((a) > (b)) ? (b) : (a))

using namespace std;

void* thread_mat_inv(void*);
void* thread_mat_mul(void*);

extern int** createRandomMatrix(unsigned, unsigned, bool);
extern double** createIdentityMatrix(unsigned);
extern void print_matrix(int**, unsigned, char*);
extern void print_matrix(double**, unsigned, char*);
extern int** createEmpyMatrix(unsigned);


int **A, **B, **C; //dopo moltiplicazione, C = A*B
double **D, **M; //dopo inversione, M = I && D = A^-1
unsigned dim;
int thread_number; // ToDo: controllare se il numero è migliorabile

static pthread_barrier_t barrier;

struct coord{
	int x;
	int y;
};

int main(int argc, char **argv){

	long min_dim, max_dim, step; 
	chrono::high_resolution_clock::time_point start, finish;
	chrono::duration<double> elapsed;
	pthread_t* threads;
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
    		print_matrix(A,dim,"A ");
    		print_matrix(B,dim,"B ");
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

      	M = new double*[dim];

      	for (int h = 0; h < dim; h++){
        	M[h] = new double[dim];
        	for (int w = 0; w < dim; w++)
                	M[h][w] = A[h][w];
      	}

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------MULTITHREAD CODE----------------------

		pthread_barrier_init(&barrier, NULL, thread_number);

		for(int i=0; i<thread_number; i++){ //one thread per column. This simplifies the algorithm greatly
			params[i].x = i; //what column we reduce
			pthread_create(&threads[i],NULL,thread_mat_inv,(void*)&params[i]);
		}

		for (int i=0; i<thread_number; i++) {
 		   pthread_join( threads[i],NULL);
		}

		//----------------------MULTITHREAD CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure
		
		if(DEBUG){
    		print_matrix(D,dim,"D ");
    		print_matrix(M,dim,"M ");
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
	coord *v = (coord*)params;
	double p;

	for(int z=0; z<2; z++){
		//riduci a triangolare superiore
		for(int k=0; k<dim; k++){ //foreach row
			pthread_barrier_wait(&barrier); //sinchro point
			p = M[k][k];
			for(int j=k+v->x; j<dim; j+=thread_number){ //foreach thread column
				M[k][j] /= p;
				D[k][j] /= p;
				for(int i=k+1;i<dim;i++){
					M[i][j] -= M[i][k]*M[k][j];
					D[i][j] -= M[i][k]*D[k][j];
				}
			}
		}

		//trasponi le matrici M e D
		for(int i=0;i<dim-1;i+=thread_number){
			M[i][i] = 1;
			for(int j=i+1; j<dim; j++){
				M[i][j]= 0;
				swap(M[i][j],M[j][i]);
				swap(D[i][j],D[j][i]);
			}
		}
	}

	pthread_exit((void*)0);
}