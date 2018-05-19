#include <iostream>
#include <chrono>
#include <cstring>
#include <omp.h>
#define DEBUG 0

using namespace std;


extern int** createRandomMatrix(long, long, bool);
extern double** createIdentityMatrix(long);
extern int** createEmpyMatrix(long);
extern void print_matrix(int**, long, char*);
extern void print_matrix(double**, long, char*);
extern void saveTimeToFile(long, double, char*);

void mat_inv(double**, double**, long);
void mat_mul(int**, int**, int**, long);

int main(int argc, char **argv){
	long min_dim, max_dim, step, dim; 
	chrono::high_resolution_clock::time_point start, finish;
	chrono::duration<double> elapsed;
	int **A, **B, **C;
  double **D, **M;
  
  omp_set_num_threads(4);
    
	//ToDo(?) si puo' mettere il path del file da salvare come argomento di input
	if(argc != 4){
    cout << "Usage: " << argv[0] << " [min_dim] [max_dim] [step]" << endl;
    return -1;
	}

	min_dim = strtol(argv[1], NULL, 10);
	max_dim = strtol(argv[2], NULL, 10)+1;
	step = strtol(argv[3], NULL, 10);


	for(dim=min_dim;dim<max_dim+1;dim+=step){

		//ToDo: parallelizzare anche questa funzione (?)
		A = createRandomMatrix(dim, dim, true); // true means "invertible"
		B = createRandomMatrix(dim, dim, false); // false means "not invertible"
    C = createEmpyMatrix(dim);
    D = createIdentityMatrix(dim);

    M = new double*[dim];

    for (int h = 0; h < dim; h++){
      M[h] = new double[dim];
      for (int w = 0; w < dim; w++)
        M[h][w] = A[h][w];
    }

    if(DEBUG){
      print_matrix(A,dim,"A ");
      print_matrix(B,dim,"B ");
    }

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------

    mat_mul(A,B,C,dim); //Moltiplico C = A*B

		//----------------------CRITICAL CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

    if(DEBUG){
      
      print_matrix(C,dim,"C ");
    }

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		if(DEBUG) cout << "MUL: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi
    saveTimeToFile(dim, elapsed.count(), "csv/multiplication_OpenMP.csv");


		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------
		
		mat_inv(M,D,dim); //Inverto M=A e metto il risultato in D

		//----------------------CRITICAL CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure


    if(DEBUG){
      print_matrix(M,dim,"M ");
      print_matrix(D,dim,"D ");
    }

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		if(DEBUG) cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi
    saveTimeToFile(dim, elapsed.count(), "csv/inversion_OpenMP.csv");

		free(A);
		free(B);
		free(C);
		free(D);
    free(M);
	}
	
	return 0;
}

//INVERSIONE
void mat_inv(double **M, double **D, long dim){
  double p;
  int i,j,k;
  for(int z=0; z<2; z++){
    //riduci a triangolare superiore
    for(k=0; k<dim; k++){ //foreach row
      p = M[k][k];
      #pragma omp parallel for private(j,i)
      for(j=k; j<dim; j++){ //foreach thread column
        M[k][j] = M[k][j]/p;
        D[k][j] = D[k][j]/p;
        for(i=k+1;i<dim;i++){
          M[i][j] -= M[i][k]*M[k][j];
          D[i][j] -= M[i][k]*D[k][j];
        }
      }
    }

    //trasponi le matrici M e D
    #pragma omp parallel for private(j,i)
    for(i=0;i<dim-1;i++){
      M[i][i] = 1;
      for(j=i+1; j<dim; j++){
        M[i][j]= 0;
        swap(M[i][j],M[j][i]);
        swap(D[i][j],D[j][i]);
      }
    }
  }

  return;
}

//MOLTIPLICAZIONE
void mat_mul(int **A,int **B, int** prodotto, long n){
    // *** Moltiplicazione Tra Matrice A*** e B*** //  
  int i,j,k;
  #pragma omp parallel for private(i,j,k)
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      prodotto[i][j] = 0;
      for (k = 0; k < n; k++) {
        prodotto[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return;
}
