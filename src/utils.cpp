#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <fstream>
#define R_MAX 10
#define eps 0.1f
/* R_MAX/2 is the biggest number that will appear in the random matrices
(outside the diagonal) 
a small R_MAX improves numerical stability when inverting a matrix */
/* eps is the maximum error allowed when checking for 
the correctness of the computed matrices */

using namespace std;

/* This file contains all the utility functions 
It could be well written, but it's not 
For example, we could have used templates  */

//The following function generates a random Matrix
float** createRandomMatrix(long height, long width, bool invertible){ 
      float** m = 0;
      int x, c;
      m = new float*[height];
      srand (time(NULL));

      if(invertible && height != width) //invertible matrix must be square
      	return 0;

      for (int h = 0, c = 0; h < height; h++, c=0){
            m[h] = new float[width];

            if(invertible){ 
            	//if diagonally dominant the matrix is invertible
            	for (int w = 0; w < width; w++)
            		if(w!=h){
            			x = rand()%R_MAX - R_MAX/2;
            			c+=abs(x);
            			m[h][w] = x;	
            		}
            	/*Condition to be diagonally dominant:
                for each h, abs(m[h][h]) > c,
                where c = (sum for every w!=h, w<dim, w++: abs(m[h][w])) */
            	m[h][h] = (float)(rand()%(R_MAX/10) + c + 1); 

            } else
            	for (int w = 0; w < width; w++)
                	m[h][w] = (float)(rand()%R_MAX - R_MAX/2);
      }

      return m;
}

//This Function creates the standard idendity matrix
float** createIdentityMatrix(long dim){
      float** m = 0;
      m = new float*[dim];

      for (int h = 0; h < dim; h++){
        m[h] = new float[dim];
        for(int l = 0; l < dim; l++)
          m[h][l] = 0;
        m[h][h] = 1;
      }

      return m;
}

//This Function creates a matrix full of 0
float** createEmptyMatrix(long dim){
      float** m = 0;
      m = new float*[dim];

      for (int h = 0; h < dim; h++){
        m[h] = new float[dim];
        for(int l = 0; l < dim; l++){
          m[h][l] = 0;
        }
      }
      return m;
}

/*Same function as before, but they create arrays instead of matrices.
  The created arrays are row-wise concatenated matrices, 
  with the same proprieties as before */
//Note: if m has 'height' rows, m[h][w] = m[h*height+w]
float* createRandomMatrixArray(long height, long width, bool invertible){
      float* m = 0;
      int x, c;
      m = new float[height*width];
      srand (time(NULL));

      if(invertible && height != width) //invertible matrix must be square
        return 0;

      for (int h = 0, c = 0; h < height; h++, c=0){

            if(invertible){ 
              //if diagonally dominant the matrix is invertible
              for (int w = 0; w < width; w++)
                if(w!=h){
                  x = rand()%R_MAX - R_MAX/2;
                  c+=abs(x);
                  m[h*height+w] = x;  
                }
              
              m[h*(height+1)] = (float)(rand()%(R_MAX/10) + c + 1);

            } else
              for (int w = 0; w < width; w++)
                  m[h*height+w] = (float)(rand()%R_MAX - R_MAX/2);
      }

      return m;
}

float* createIdentityMatrixArray(long dim){
      float* m = 0;
      m = new float[dim*dim];

      for (int h = 0; h < dim; h++){
        for(int j = 0; j < dim; j++)
          m[h*dim+j] = 0;
        m[h*(dim+1)] = 1;
      }

      return m;
}

float* createEmptyMatrixArray(long dim){
      float* m = 0;
      m = new float[dim*dim];
      for (int h = 0; h < dim; h++){
        for(int j = 0; j < dim; j++)
          m[h*dim+j] = 0;
      }
      return m;
}

/*The following function saves the dim (x) and the recorded time (y)
  to a file (*filename) in the format: x y */
void saveTimeToFile(long x, float y, char* filename){
  ofstream file;
  file.open(filename, ios_base::app);
  file << x << " " << y << endl;
  file.close();
}
//Obviously print the matrix 
void print_matrix(float** A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i][j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}
//Print the arrais as a matrix
void print_array_as_matrix(float* A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i*n+j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}
//Print the arrais as a Reverse Matrix
void print_array_as_matrixT(float* A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[j*n+i] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}
//Check if Multiplied Matrix is correct 
bool multipliedMatrixIsCorrect(float **A, float **B, float **C, long dim){
  
  float** check = 0;
  check = new float*[dim];

  for (int i = 0; i < dim; i++){
    check[i] = new float[dim];
    for(int j = 0; j < dim; j++){
      check[i][j] = 0;
      for (int k = 0; k < dim; k++) {
        check[i][j] += A[i][k] * B[k][j];
      }
      if(fabs(check[i][j]-C[i][j])>eps){
        cout << "Error detected on indexes: " << i << " " << j << endl; 
        free(check);
        return false;
      }
    }
  }
  free(check);
  return true;

}
//Check if the multiplied Matrix in the Cuda language is correct
bool multipliedMatrixCudaIsCorrect(float *A, float *B, float *C, long dim){
  
  float* check = 0;
  check = new float[dim*dim];

  for (int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      check[i*dim+j] = 0;
      for (int k = 0; k < dim; k++) {
        check[i*dim+j] += A[i*dim+k] * B[k*dim+j];
      }
      if(fabs(check[i*dim+j]-C[i*dim+j])>eps){
        cout << "Error detected on indexes: " << i << " " << j << endl;
        free(check);
        return false;
      }
    }
  }
  free(check);
  return true;

}
/*Check if the multiplied matrix calculated with the libraries
in the Cuda language is correct */
bool multipliedMatrixCublasIsCorrect(float *A, float *B, float *C, long dim){
  
  float* check = 0;
  check = new float[dim*dim];

  for (int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      check[j*dim+i] = 0;
      for (int k = 0; k < dim; k++) {
        check[j*dim+i] += A[k*dim+i] * B[j*dim+k];
      }
      if(fabs(check[j*dim+i]-C[j*dim+i])>eps){
        cout << "Error detected on indexes: " << i << " " << j << endl;
        free(check);
        return false;
      }
    }
  }
  free(check);
  return true;

}