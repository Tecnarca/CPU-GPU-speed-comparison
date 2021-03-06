#include <iostream>
#include <chrono>
#include <cstring>
#include <omp.h>
#define DEBUG 0
//If DEBUG is setted, the program will print the used matrices and the times on the stdout

using namespace std;

/* From utils.cpp */
extern float** createRandomMatrix(long, long, bool);
extern float** createIdentityMatrix(long);
extern float** createEmptyMatrix(long);
extern void print_matrix(float**, long, char*);
extern void saveTimeToFile(long, float, char*);
extern bool multipliedMatrixIsCorrect(float**, float**, float**, long);

/* For the inversion and the multiplication of the matrices, the execution time of these functions will be measured */
void mat_inv(float**, float**, long);
void mat_mul(float**, float**, float**, long);

int main(int argc, char **argv){
  long min_dim, max_dim, step, dim; //Used to determine what matrix dimensions we will test
  chrono::high_resolution_clock::time_point start, finish; //Used to implement time measurement
  chrono::duration<double> elapsed; //Will contain the elapsed time
  float **A, **B, **C; //After moltiplication, C=A*B
  float **D, **M; //M=A, D=Identity and after inversion: D = A^-1, M=Identity
  
  omp_set_num_threads(4); //Set the number of threads to 4, if you want you can change this

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

		A = createRandomMatrix(dim, dim, true); // true means "invertible"
		B = createRandomMatrix(dim, dim, false); // false means "not invertible"
    C = createEmptyMatrix(dim);
    D = createIdentityMatrix(dim);

    //M=A
    M = new float*[dim];
    for (int h = 0; h < dim; h++){
      M[h] = new float[dim];
      for (int w = 0; w < dim; w++)
        M[h][w] = A[h][w];
    }

    if(DEBUG){
      print_matrix(A,dim,"A ");
      print_matrix(B,dim,"B ");
    }

    //BEGIN MATRICES MULTIPLICATION

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------

    mat_mul(A,B,C,dim); //Multiply C = A * B

		//----------------------CRITICAL CODE----------------------

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

    saveTimeToFile(dim, elapsed.count(), "csv/multiplication_OpenMP.csv");

    //BEGIN MATRIX INVERSION

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------
		
		mat_inv(M,D,dim); //Invert M = A and put the result in D

		//----------------------CRITICAL CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure


    if(DEBUG){
      print_matrix(M,dim,"M ");
      print_matrix(D,dim,"D ");
      bool correct = multipliedMatrixIsCorrect(A,D,M,dim);
      if(!correct){
        cout << "Multiplied matrix is not correct, aborting..." << endl;
        return -1;
      }
    }

		elapsed = finish - start; //compute time difference

		if(DEBUG) cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;

    saveTimeToFile(dim, elapsed.count(), "csv/inversion_OpenMP.csv");

    //Free because we will reallocate memory in the next for step
		free(A);
		free(B);
		free(C);
		free(D);
    free(M);
	}
	
	return 0;
}

//INVERSION
void mat_inv(float **M, float **D, long dim){
  double s;
  int i,j;

    //reduce M to upper triangular 
    for(int piv=0; piv<dim; piv++){ //for each row
      #pragma omp parallel for private(i,j,s)
      for(int i=piv+1; i<dim; i++){ //for each column
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
      #pragma omp parallel for private(i,j,s)
      for(int i=0; i<piv; i++){ //for each column
        s = (double)M[i][piv]/M[piv][piv];
        for(int j=0;j<dim;j++){ //for every element
          D[i][j] -= (float)s*D[piv][j];
          if(j<=piv) M[i][j] -= s*M[piv][j];
        }
      }
    }
    
    //scales down the matrix
    for(int i=0;i<dim;i++){
      s = M[i][i];
      #pragma omp parallel for private(j)
      for(int j=0;j<dim;j++){
        D[i][j]/=s;
        M[i][j]/=s;
      }
    }

  return;
}

//MOLTIPLICATION
void mat_mul(float **A,float **B, float** prodotto, long n){

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
