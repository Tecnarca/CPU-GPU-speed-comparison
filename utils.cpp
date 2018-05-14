#include <ctime>
#include <cstdlib>
#include <cmath>

/* in questo file vanno tutte le funzioni non di interesse per il progetto in sè */

/*sono abbastanza sicuro che questo codice funzioni ma non ne ho la certezza, è da testare*/
int** createRandomMatrix(unsigned height, unsigned width, bool invertible)
    {
      int** m = 0;
      int x, c;
      m = new int*[height];
      srand (time(NULL));

      if(invertible && height != width) //invertible matrix must be square
      	return 0;

      for (int h = 0; h < height; h++){
            m[h] = new int[width];

            if(invertible){ 
            	//diagonally dominant
            	for (int w = 0, c = 0; w < width; w++)
            		if(w!=h){
            			x = rand()%1000 - 500;
            			c+=abs(x);
            			m[h][w] = x;	
            		}
            	
            	m[h][h] = rand() + c;
            	c=0;

            } else
            	for (int w = 0; w < width; w++)
                	m[h][w] = rand()%1000 - 500;
      }

      return m;
    }