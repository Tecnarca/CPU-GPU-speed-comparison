# CPU-GPU-speed-comparison
A simple comparison between single thread program against multi-threading and CUDA, through Matrix multiplication and Matrix inversion.

Project of Electronic calculators and computer networks course held by A. Mancini at UNIVPM.

The programs executes a Matrix multiplication and Matrix inversion storing the execution times of operations.

On CUDA the programs stores also the transfer time of data from CPU and GPU and the return time.

With a Python script, all execution and transfer time are plotted.

On the x-axis there are Matrix size, on the y-axis there are times.

# How to compile programs

Before compiling, you might want to set the `DEBUG` variable in any cpp or cu file to 1, this will show the various matrices used during the program execution.

All the commands to compile the programs are wirtten in:

>compiler.sh

Further instructions on how the programs were compiled, can be found inside that file. To run it, execute the followin in a terminal:

```
chmod +x compiler.sh
./compiler.sh
```
The executable files will be in the `bin/` folder.

# How to run the programs

We provide the latest compiled versions of the programs, with the debug variable unset.
To use the compiled version of the single files, located in `bin/`, you might want to check for missing libraries by running the command:

`ldd bin/*`

Or it's possible to check a specific executable (ex: FILENAME), running the command:

`ldd bin/FILENAME`

If you have all the required libraries, you can run every program using the `run_all.sh` script. You must pass to the script three integer parameters: 
> SMALLER_MATRIX_SIZE: the smallest matrix the programs will multiply and invert
> BIGGEST_MATRIX_SIZE: the biggest matrix the programs will multiply and invert
> GROWING_STEP: the programs will keep adding this number to SMALLER_MATRIX_SIZE and executing multiplication and inversion, until it reaches BIGGEST_MATRIX_SIZE

You can run the programs with:

```
chmod +x run_all.sh
./compiler.sh [SMALLER_MATRIX_SIZE] [BIGGEST_MATRIX_SIZE] [GROWING_STEP]
```
You can found further instructions on how to run single programs inside the `run_all.sh` script.

If no `DEBUG` variable is setted, almost no output will show. At the end of the execution, you will have a `csv/` folder, containing several files that are used to track the executing times of each code section, as described below. 

# graph_plot.py

To be contiued... (spiegare come funziona graph plot e mostrare il plot che abbiamo trovato noi)


