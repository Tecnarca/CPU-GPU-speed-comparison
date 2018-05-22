# CPU-GPU-speed-comparison
A simple comparison between single thread program against multi-threading and CUDA, through Matrix multiplication and Matrix inversion.
Project of Electronic calculators and computer networks course held by A. Mancini at UNIVPM 

The programs executes a Matrix multiplication and Matrix inversion storing the execution times of operations.
On CUDA the programs stores also the transfer time of data from CPU and GPU and the return time.
With a Python script, all execution and transfer time are plotted. 
On the x-axis there are Matrix size, on the y-axis there are times.

 #monocore.cpp

 #multicore.cpp

 #cuda.cpp

If you are using the compiled version of the files, you can check what libraries you're missing by running the command
ldd bin/*
If you want to check a specific file (let's say, FILENAME), use
ldd bin/FILENAME
