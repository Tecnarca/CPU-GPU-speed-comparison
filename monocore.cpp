#include <iostream>
#include <chrono>
#include <cstring>
#define DEBUG 1

using namespace std;

//INVERSIONE
double** mat_inv(int **A,unsigned n){
double **I;
I=new double*[n];
for(int i=0;i<n;i++) I[i]=new double[n];
// *** Popolazione della matrice identità  *** //
for(int i=0;i<n;i++)
  for(int j=0;j<n;j++)
    if(i==j) I[i][j] = 1;
    else I[i][j]=0;
// *** Stampa della matrice A*** //
cout << "\n***** MATRICE A ******\n\n";
for(int i=0;i<n;i++){
  for(int j=0;j<n;j++)
    cout << A[i][j] << "\t";
  cout << "\n"; 
}
cout << "*********************\n\n";

// *** Costruzione della matrice [A|I] *** //
double **B;
B=new double*[n];
for(int i=0;i<n;i++) B[i]=new double[2*n];

for(int i=0;i<n;i++)
  for(int j=0;j<n;j++)
    B[i][j]=A[i][j];

int k=0;
for(int i=0;i<n;i++) {
  for(int j=n;j<2*n;j++,k++)
    B[i][j]=I[i][k];
  k=0;
}

// *** Eliminazione sotto la diagonale principale *** //
double *tmp; tmp=new double[2*n];
for(int j=0;j<n-1;j++)
  for(int i=j+1;i<n;i++)
    if(B[i][j]!=0) {
      double mol=B[i][j]/B[j][j];
      for(int k=0;k<2*n;k++) tmp[k]=mol*B[j][k];
      for(int k=0;k<2*n;k++) B[i][k]-=tmp[k];
    }

// *** Eliminazione sopra la diagonale principale *** //
for(int j=n-1;j>0;j--)
  for(int i=j-1;i>=0;i--)
    if(B[i][j]!=0) {
      double mol=B[i][j]/B[j][j];
      for(int k=0;k<2*n;k++) tmp[k]=mol*B[j][k];
      for(int k=0;k<2*n;k++) B[i][k]-=tmp[k];
    }

// *** Ultimo step per ottenere la matrice a blocchi [I|A] *** //
for(int i=0;i<n;i++)
  if(B[i][i]!=1) {
    double mol=B[i][i];
    for(int k=0;k<2*n;k++)
      B[i][k]=B[i][k]/mol;
  }

// *** Copia dell'inversa ottenuta *** //
double** Inv;
Inv=new double*[n];
for(int i=0;i<n;i++) Inv[i]=new double[n];
k=0;
for(int i=0;i<n;i++) {
  for(int j=n;j<2*n;j++,k++)
    Inv[i][k]=B[i][j];
  k=0;
}

// *** Stampa della matrice Inversa*** //
cout << "\n***** MATRICE INVERSA ******\n\n";
for(int i=0;i<n;i++){
  for(int j=0;j<n;j++)
    cout << Inv[i][j] << "\t";
  cout << "\n"; 
}
cout << "***************************\n\n";

// *** Azzeramento della matrice idendità utilizzata prima, che utilizzeremo in seguito come supporto *** //
for(int i=0;i<n;i++)
  for(int j=0;j<n;j++)
     I[i][j]=0;

for(int i=0;i<n;i++)
  for(int j=0;j<n;j++)
    for(k=0;k<n;k++)
      I[i][j]+=(A[i][k]*Inv[k][j]);
 


/*// *** Stampa della matrice identità  ottenuta da A*A^-1 *** //
cout  << "\n** Stampa di A*A^-1 **\n";
for(int i=0;i<n;i++) {
  for(int j=0;j<n;j++)
    cout << I[i][j] << "\t";
  cout << "\n";
}
*/
cout << "***********************\n";
return Inv;
}; 
//MOLTIPLICAZIONE
int** mat_mul(int **A,int **B,unsigned n){
int **prodotto = new int*[n];
for(int i=0;i<n;i++) prodotto[i]=new int[n];
// *** Stampa della matrice A*** //
  cout << "\n***** MATRICE A ******\n\n";
    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << A[i][j] << "\t";
      cout << "\n"; 
    }
  cout << "*********************\n\n";
// *** Stampa della matrice B*** //
  cout << "\n***** MATRICE B ******\n\n";
    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++)
        cout << B[i][j] << "\t";
      cout << "\n"; 
    }
  cout << "*********************\n\n";
// *** Moltiplicazione Tra Matrice A*** e B*** //  
  int i,j,k;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        prodotto[i][j] = 0;
          for (k = 0; k < n; k++) {
            prodotto[i][j] += A[i][k] * B[k][j];
          }
      }
    }
// *** Stampa della matrice prodotto*** //
  cout << "\n***** MATRICE PRODOTTO ******\n\n";
    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
         cout << prodotto[i][j] <<"\t";
      }
      cout << '\n'; // o cout << end;
    }
return prodotto;
};
extern int** createRandomMatrix(unsigned, unsigned, bool);

int main(int argc, char **argv){

	unsigned dim;
	long min_dim, max_dim, step; 
	chrono::high_resolution_clock::time_point start, finish;
	chrono::duration<double> elapsed;
	int **A, **B, **C;
    double **D;
    

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

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------

		D = (double**)mat_inv(A,dim); //Inverto A

		//----------------------CRITICAL CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		cout << "INV: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi

		start = chrono::high_resolution_clock::now(); //start time measure

		//----------------------CRITICAL CODE----------------------
		
		C = (int**)mat_mul(A,B,dim); //Moltiplico A*B, 

		//----------------------CRITICAL CODE----------------------

		finish = chrono::high_resolution_clock::now(); //end time measure

		elapsed = finish - start; //compute time difference

		//ToDo: output to file instead of console
		//format of the output to file: DECIDE CHI USA MATPLOTLIB
		cout << "MUL: With dimension " << dim << ", elapsed time: " << elapsed.count() << " s" << endl;
		//elapsed.count() restituisce il tempo in secondi
	
		free(A);
		free(B);
		free(C);
		free(D);
	}
	
	return 0;
}