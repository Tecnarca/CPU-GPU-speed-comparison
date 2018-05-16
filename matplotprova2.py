import numpy as np
from io import StringIO 
import matplotlib.pyplot as plt

filename="/home/mamb4comm/Scrivania/esempio.txt"
with open(filename) as f:
    data = f.read()

data = data.split('\n\n')

labels= ['Monocore', 'Multicore', 'CUDA']
i=0
for d in data:
    ds = np.loadtxt(StringIO(str(d)))
    plt.plot(ds[:,0],ds[:,1], label=labels[i])
    i=i+1
    
plt.legend(loc='upper left')
plt.xlabel("Matrix size")
plt.ylabel("Time")
plt.show()

