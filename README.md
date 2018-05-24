# CPU-GPU-speed-comparison
A simple comparison between single thread program against multi-threading and CUDA, through Matrix multiplication and Matrix inversion.

Project of Electronic calculators and computer networks course held by A. Mancini at UNIVPM.

The programs executes a Matrix multiplication and Matrix inversion storing the execution times of operations.

On CUDA the programs stores also the transfer time of data from CPU and GPU and the return time.

With a Python script, all execution and transfer time are plotted.

On the x-axis there are Matrix size, on the y-axis there are times.

# How to compile programs

All the commands to compile and run the programs are wirtten in:

>compile.sh

Where is possible to find all the command to compile each program with the relative libraries and all the useful tips to run the file correctly.

For example:

```
mkdir -p objects/
...
nvcc -c src/cublas.cu -o objects/cublas.o -std=c++11 -w
...
g++ -c src/utils.cpp -o objects/utils.o -std=c++11
```
To use the compiled version of the files, it's possible to check the missing libraries by running the command:

`ldd bin/*`

Or it's possible to check a specific file (ex: FILENAME), running the command:

`ldd bin/FILENAME`

# graph_plot.py

To be contiued...


