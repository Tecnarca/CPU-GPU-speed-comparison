# CPU-GPU-speed-comparison
A simple comparison between single thread program against multi-threading and CUDA, through Matrix multiplication and Matrix inversion.

Project of Electronic calculators and computer networks course held by A. Mancini at UNIVPM.

The programs executes a Matrix multiplication and Matrix inversion storing the execution times of operations.

On CUDA the programs stores also the transfer time of data from CPU and GPU and the return time.

With a Python script, all execution and transfer time are plotted.

On the x-axis there are Matrix size, on the y-axis there are times.

![Plotted data](curves.png?raw=true "Plots")

More information on how to read the data plots can be found below, in the `Plotting the Data` section.

## How to compile programs

Before compiling, you might want to set the `DEBUG` variable in any cpp or cu file to 1, this will show the various matrices used during the program execution.

All the commands to compile the programs are wirtten in:

>compiler.sh

Further instructions on how the programs were compiled, can be found inside that file. To run it, execute the following in a terminal:

```
chmod +x compiler.sh
./compiler.sh
```
The executable files will be in the `bin/` folder.

## How to run the programs

We provide the latest compiled versions of the programs, with the debug variable unset.
To use the compiled version of the single files, located in `bin/`, you might want to check for missing libraries by running the command:

`ldd bin/*`

Or it's possible to check a specific executable (ex: FILENAME), running the command:

`ldd bin/FILENAME`

If you have all the required libraries, you can run every program using the `run_all.sh` script. You must pass to the script three integer parameters: 
* SMALLER_MATRIX_SIZE: the smallest square matrix of size SMALLER_MATRIX_SIZExSMALLER_MATRIX_SIZE the programs will multiply and invert
* BIGGEST_MATRIX_SIZE: the biggest square matrix of size BIGGEST_MATRIX_SIZExBIGGEST_MATRIX_SIZE the programs will multiply and invert
* GROWING_STEP: the programs will keep adding this number to SMALLER_MATRIX_SIZE and executing multiplication and inversion, until it reaches BIGGEST_MATRIX_SIZE

You can run the programs with:

```
chmod +x run_all.sh
./compiler.sh [SMALLER_MATRIX_SIZE] [BIGGEST_MATRIX_SIZE] [GROWING_STEP]
```

For example:

```
chmod +x run_all.sh
./compiler.sh 100 1400 100
```


You can found further instructions on how to run single programs inside the `run_all.sh` script.

If no `DEBUG` variable is setted, almost no output will show. At the end of the execution, you will have a `csv/` folder, containing several files that are used to track the executing times of each code section, as described below. 

## Plotting the data

By running the `graph_plot.py` script with the command

```
python graph_plot.py
```

You can plot the data stored inside the `csv/` folder. More information on how to edit this program can be found in the file itself.

This will show many plots, each one with matrix dimension (only the width is shown, as they are square matrices) on the x-axis and times (in seconds) on y-axis:

* The `multiplication_` and `inversion_` ones shows how much time every program spent computing the matrices inversion and multiplication
* The `multiplication_CU` and `inversion_CU` ones shows how much time CUDA (the 'DA' line) and CUBLAS (the 'BLAS' line) programs spent in computing the matrices inversion and multiplication
*The four graphs on the second row, shows how much time of the total computation was spent for passing the matrices values to the GPU (`load_multiplication_` and `load_inversion_`) and to read the resulting matrices back (`read_multiplication_` and `load_inversion_`)

The plots shown above was generated running this project on Debian Wheezy with an Intel Core i5 4690 Quad core 3.5GHz CPU and nVidia GeForce GTX970 GPU

## How times has been measured

We tried to confront different approaches for matrix operations, but because the GPU needs to hold the data in his own RAM, it was not sufficent to measure only the computation time to generate the output matrix, but also the transfer times from the system RAM to the GPU and the copyback (GPU to RAM) times. It was right to emphasize transfer times from CPU to GPU beacuse they produce a remarkable difference in the output times of multiplied and inverted matrices.

We have timed with CUDA events the following instructions in `cuda.cu` and `cublas.cu` :


>cuda.cu


To copy the matrices A and B from RAM to GPU RAM

```
status = cudaMemcpy(gpu_inv_A, M, data_size, cudaMemcpyHostToDevice);
status = cudaMemcpy(gpu_inv_I, D, data_size, cudaMemcpyHostToDevice);
cudaDeviceSynchronize();
```

To copy back matrices from GPU RAM to RAM 

```
status = cudaMemcpy(M, gpu_inv_A, data_size, cudaMemcpyDeviceToHost);
status = cudaMemcpy(D, gpu_inv_I, data_size, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
```



>cublas.cu



To copy matrices A and B from RAM to GPU RAM

```
status = cudaMemcpy(gpu_A, A, data_size,cudaMemcpyHostToDevice); //copy gpu_A <-A
status = cudaMemcpy(gpu_D, D, data_size,cudaMemcpyHostToDevice); //copy gpu_D <-D
cusolverStatus = cusolverDnSgetrf_bufferSize(cuhandle,dim,dim,gpu_A,dim,&Lwork);
```

And copy back to GPU RAM to RAM

```
status = cudaMemcpy (&info_gpu , gpu_info , sizeof(int), cudaMemcpyDeviceToHost );  
status = cudaMemcpy(C, gpu_D , dim*dim*sizeof(float), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
```
Aside from these particular steps for the GPU, every program used exactly the same algorithm for multiplicating and inverting (gauss jordan, implemented in 3 steps: the "upper triangular reduction", "low triangular reduction" and "scaling"), only written using the different approaches, except for CUBLAS, that uses an LU factorization and solves the generic system AxB=I). 
So, every other timed portion (timed with CHRONO for the CPU and CUDA events for the GPU) issues only the instructions that produces the output. For example, in `cuda.cu` and `multithread.cpp`:

>cuda.cu

```
for(int i=0;i<dim-1;i++)
  upperReduction <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim, i);
for(int i=dim-1;i>0;i--)
  lowerReduction <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim, i);
scale <<< blocksPerGrid, threadsPerBlock >>> (gpu_inv_A, gpu_inv_I, dim);
```

>multithread.cpp

```
for(int i=0; i<thread_number; i++){
	params[i].x = i*dim/thread_number;
	params[i].y = MIN((i+1)*dim/thread_number,dim);
	pthread_create(&threads[i],NULL,thread_mat_mul,(void*)&params[i]);
}

for (int i=0; i<thread_number; i++)
	pthread_join(threads[i],NULL);
```
A more detailed explanation of the timing was realized can be found inside the various files.

## Built with
* [CUDA 9.1](https://developer.nvidia.com/cuda-toolkit) - The toolkit to write and executing programs on nVidia GPUs
* [OpenMP](https://www.openmp.org/) - API to write simple multithread programs
* [pThread](http://man7.org/linux/man-pages/man7/pthreads.7.html) - The standard Linux multithread API




