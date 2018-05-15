#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#define R_MAX 1000

using namespace std;

/* in questo file vanno tutte le funzioni non di interesse per il progetto in sè */

int** createRandomMatrix(unsigned height, unsigned width, bool invertible){
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
            	
            	m[h][h] = rand()%(R_MAX/10) + c;

            } else
            	for (int w = 0; w < width; w++)
                	m[h][w] = rand()%R_MAX - R_MAX/2;
      }

      return m;
}

double** createIdentityMatrix(unsigned dim){
      double** m = 0;
      m = new double*[dim];

      for (int h = 0; h < dim; h++){
        m[h] = new double[dim];
        m[h][h] = 1;
      }

      return m;
}

int** createEmpyMatrix(unsigned dim){
      int** m = 0;
      m = new int*[dim];

      for (int h = 0; h < dim; h++){
        m[h] = new int[dim];
        m[h][h] = 1;
      }
      return m;
}

void print_matrix(int** A, unsigned n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i][j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}

void print_matrix(double** A, unsigned n, char* s){
  cout << "\n***** MATRICE " << s << "******\n\n";
  for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i][j] << "\t";
      cout << endl;
  }
  cout << "*********************\n\n"; 
}