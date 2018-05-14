#include <iostream>
#include <pthread.h>
#include <chrono>
#include <cstring>
#define DEBUG 1

using namespace std;

void** thread_mat_inv(); //header e corpo da implementare
void** thread_mat_mul(); //header e corpo da implementare
extern int** createRandomMatrix(unsigned, unsigned, bool);

int main(int argc, char **argv){

	unsigned dim;
	long min_dim, max_dim, step; 
	chrono::high_resolution_clock::time_point start, finish;
	chrono::duration<double> elapsed;
	int **A, **B, **C;

	//ToDo(?) si puo' mettere il path del file da salvare come argomento di input
	if(argc != 4){
		cout << "Usage: " << argv[0] << " [min_dim] [max_dim] [step]" << endl;
	}
	
	min_dim = strtol(argv[1], NULL, 10);
	max_dim = strtol(argv[2], NULL, 10)+1;
	step = strtol(argv[3], NULL, 10);


	for(dim=min_dim;dim<max_dim;dim+=step){

		//ToDo: parallelizzare anche questa funzione (?)
		A = createRandomMatrix(dim, dim, true); // 1 means "invertible"
		B = createRandomMatrix(dim, dim, false); // 0 means "not invertible"

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------MULTITHREAD CODE----------------------

		C = (int**)thread_mat_mul(); //moltiplicare A e B

		//----------------------MULTITHREAD CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		cout << "MUL: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------MULTITHREAD CODE----------------------
		
		C = (int**)thread_mat_inv(); //invertire A e fate quello che volete

		//----------------------MULTITHREAD CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi
				
		free(A);
		free(B);
	}
	
	return 0;
}

