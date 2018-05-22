#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <fstream>
#define R_MAX 20
/* R_MAX/2 is the biggest number that will appear in the random matrices */
/* a small R_MAX improves numerical stability when inverting a matrix */


using namespace std;

/* This file contains all the utility functions */
/* It could be well written, but it's not */
/* We will write this better in the future */

int** createRandomMatrix(long height, long width, bool invertible){ 
      int** m = 0;
      int x, c;
      m = new int*[height];
      srand (time(NULL));

      if(invertible && height != width) //invertible matrix must be square
      	return 0;

      for (int h = 0, c = 0; h < height; h++, c=0){
            m[h] = new int[width];

            if(invertible){ 
            	//diagonally dominant == surely invertible
            	for (int w = 0; w < width; w++)
            		if(w!=h){
            			x = rand()%R_MAX - R_MAX/2;
            			c+=abs(x);
            			m[h][w] = x;	
            		}
            	// Condition to be diagonally dominant: for each h, abs(m[h][h]) > c, where c = (sum for every w!=h, w<dim, w++: abs(m[h][w]))
            	m[h][h] = rand()%(R_MAX/10) + c + 1; 

            } else
            	for (int w = 0; w < width; w++)
                	m[h][w] = rand()%R_MAX - R_MAX/2;
      }

      return m;
}

//This creates the standard idendity matrix
double** createIdentityMatrix(long dim){
      double** m = 0;
      m = new double*[dim];

      for (int h = 0; h < dim; h++){
        m[h] = new double[dim];
        m[h][h] = 1;
      }

      return m;
}

//matrix full of 0
int** createEmpyMatrix(long dim){
      int** m = 0;
      m = new int*[dim];

      for (int h = 0; h < dim; h++){
        m[h] = new int[dim];
        for(int l = 0; l < dim; l++){
          m[h][l] = 0;
        }
      }
      return m;
}

//Same function as before, but they create arrays instead of matrices.
//The created arrays are row-wise concatenated matrices, with the same proprieties as before
//Note: if m has 'height' rows, m[h][w] = m[h*height+w]
int* createRandomMatrixArray(long height, long width, bool invertible){
      int* m = 0;
      int x, c;
      m = new int[height*width];
      srand (time(NULL));

      if(invertible && height != width) //invertible matrix must be square
        return 0;

      for (int h = 0, c = 0; h < height; h++, c=0){

            if(invertible){ 
              //diagonally dominant == surely invertible
              for (int w = 0; w < width; w++)
                if(w!=h){
                  x = rand()%R_MAX - R_MAX/2;
                  c+=abs(x);
                  m[h*height+w] = x;  
                }
              
              m[h*(height+1)] = rand()%(R_MAX/10) + c + 1;

            } else
              for (int w = 0; w < width; w++)
                  m[h*height+w] = rand()%R_MAX - R_MAX/2;
      }

      return m;
}

double* createIdentityMatrixArray(long dim){
      double* m = 0;
      m = new double[dim*dim];

      for (int h = 0; h < dim; h++){
        m[h*(dim+1)] = 1;
      }

      return m;
}

int* createEmpyMatrixArray(long dim){
      int* m = 0;
      m = new int[dim*dim];
      for (int h = 0; h < dim; h++){
        for(int j = 0; j < dim; j++)
          m[h*dim+j] = 0;
      }
      return m;
}

//Saves the dim (x) and the recorded time (y) to a file (*filename) in the format:
// x y
void saveTimeToFile(long x, double y, char* filename){
  ofstream file;
  file.open(filename, ios_base::app);
  file << x << " " << y << endl;
  file.close();
}

//Print functions
void print_matrix(int** A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i][j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}

void print_matrix(double** A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i][j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}

void print_array_as_matrix(int* A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i*n+j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}

void print_array_as_matrix(double* A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i*n+j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}

void print_array_as_matrix(float* A, long n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i*n+j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}