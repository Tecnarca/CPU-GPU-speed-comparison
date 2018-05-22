#include <iostream>
#include <chrono>
#include <cstring>
#define DEBUG 0
//If DEBUG is setted, will print the used matrices and the times on the stdout

using namespace std;

/* From utils.cpp */
extern int** createRandomMatrix(long, long, bool);
extern double** createIdentityMatrix(long);
extern int** createEmpyMatrix(long);
extern void print_matrix(int**, long, char*);
extern void print_matrix(double**, long, char*);
extern void saveTimeToFile(long, double, char*);

/* For inverting and multiplicating matrices, these functions will be timed */
void mat_inv(double**, double**, long);
void mat_mul(int**, int**, int**, long);

int main(int argc, char **argv){
	long min_dim, max_dim, step, dim; //Used to determine what matrix dimensions we will test
	chrono::high_resolution_clock::time_point start, finish; //Used to implement the timing
	chrono::duration<double> elapsed; //Will contain the elapsed time
	int **A, **B, **C; //After multiplicating, C=A*B
  double **D, **M; //M=A, D=Identity and after inversion: D = A^-1, M=Identity
    
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

		A = createRandomMatrix(dim, dim, true); // true means "invertible"
		B = createRandomMatrix(dim, dim, false); // false means "not invertible"
    C = createEmpyMatrix(dim);
    D = createIdentityMatrix(dim);

    //M = A
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

    //BEGIN MATRICES MULTIPLICATION

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------

    mat_mul(A,B,C,dim); //C = A*B

		//----------------------CRITICAL CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

    if(DEBUG){
      
      print_matrix(C,dim,"C ");
    }

		elapsed = finish - start; //compute time difference

    //elapsed.count() gives the time in seconds
		if(DEBUG) cout << "MUL: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;

    saveTimeToFile(dim, elapsed.count(), "csv/multiplication_SingleThread.csv");

    //BEGIN MATRIX INVERSION

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------
		
		mat_inv(M,D,dim); //Invert M, result is in D

		//----------------------CRITICAL CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure


    if(DEBUG){
      print_matrix(M,dim,"M ");
      print_matrix(D,dim,"D ");
    }

		elapsed = finish - start; //compute time difference

		if(DEBUG) cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;

    saveTimeToFile(dim, elapsed.count(), "csv/inversion_SingleThread.csv");

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
void mat_inv(double **M, double **D, long dim){
  double p;
  for(int z=0; z<2; z++){ //done two times:
    //reduce M to upper triangular 
    for(int k=0; k<dim; k++){ //foreach row
      p = M[k][k];
      for(int j=k; j<dim; j++){ //foreach column
        M[k][j] = M[k][j]/p;
        D[k][j] = D[k][j]/p;
        for(int i=k+1;i<dim;i++){ //for every element
          M[i][j] -= M[i][k]*M[k][j];
          D[i][j] -= M[i][k]*D[k][j];
        }
      }
    }

    //traspose M and D, the whole function could be made faster
    //by playing with the indexes instead of reducing the matrix two times
    for(int i=0;i<dim-1;i++){
      M[i][i] = 1;
      for(int j=i+1; j<dim; j++){
        M[i][j]= 0;
        swap(M[i][j],M[j][i]);
        swap(D[i][j],D[j][i]);
      }
    }
  }

  return;
}

//MOLTIPLICATION
void mat_mul(int **A,int **B, int** prodotto, long n){
 
  int i,j,k;
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
