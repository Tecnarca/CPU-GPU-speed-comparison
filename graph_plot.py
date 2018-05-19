import numpy as np
from io import StringIO
import os
import glob
import matplotlib.pyplot as plt

#Files are searched inside the directory "csv" that must be inside the current directory

i=0
types = ['multiplication_*.csv', 'inversion_*.csv', 'multiplication_CU*.csv', 'inversion_CU*.csv', 'load_multiplication_*.csv', 'read_multiplication_*.csv', 'load_inversion_*.csv', 'read_inversion_*.csv']
for mode in types:
	i=i+1
	for file in glob.glob(os.path.join(os.getcwd(),"csv",mode)):
		with open(file) as f:
		    data = f.read()
		data = data.split('\n\n') 
		ds = np.loadtxt(StringIO(str(data[0])))
		plt.subplot(2, 4, i)
		plt.plot(ds[:,0],ds[:,1],label=file[len(os.path.join(os.getcwd(),"csv",mode))-5:-4])
	plt.title(mode[:-5])
	plt.legend(loc='upper left')
	plt.xlabel("Matrix size")
	plt.ylabel("Time")
plt.show()