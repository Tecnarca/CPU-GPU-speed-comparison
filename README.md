# CPU-GPU-speed-comparison
A simple comparison between single thread program against multi-threading and CUDA, through Matrix multiplication and Matrix inversion.

Project of Electronic calculators and computer networks course held by A. Mancini at UNIVPM.

The programs executes a Matrix multiplication and Matrix inversion storing the execution times of operations.

On CUDA the programs stores also the transfer time of data from CPU and GPU and the return time.

With a Python script, all execution and transfer time are plotted.

On the x-axis there are Matrix size, on the y-axis there are times.

## How to compile programs

Before compiling, you might want to set the `DEBUG` variable in any cpp or cu file to 1, this will show the various matrices used during the program execution.

All the commands to compile the programs are wirtten in:

>compiler.sh

Further instructions on how the programs were compiled, can be found inside that file. To run it, execute the followin in a terminal:

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

This is the plots we generated running this project on Debian Wheezy with an Intel Core i5 4690@3.5GHz CPU and nVidia GeForce GTX970 GPU:

![Plotted data](curves.png?raw=true "Plots")

## Built with
* [CUDA 9.1](https://developer.nvidia.com/cuda-toolkit) - The toolkit to write and executing programs on nVidia GPUs
* [OpenMP](https://www.openmp.org/) - API to write simple multithread programs
* [pThread](http://man7.org/linux/man-pages/man7/pthreads.7.html) - The standard Linux multithread API




