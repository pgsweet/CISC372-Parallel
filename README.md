# Parallel Computing Work
This is a repo of my classwork for my parallel computing class during spring 2025 at UD

## How to Run

#### On Windows

General command to compile and run
```console
cc -lm -g <file name>; ./a.exe
```
Running a given example
```console
cc -lm -g given_examples/<file name>; ./a.exe
```
Running a HW file
```console
cc -lm -g HW/<file name>; ./a.exe
```
Checking for memory leaks with valgrind
```console
valgrind --leak-check=yes ./a.exe
```

#### On Linux
General command to compile and run
```console
cc -lm -g <file name>; ./a.out
```
Running a given example
```console
cc -lm -g given_examples/<file name>; ./a.out
```
Running a HW file
```console
cc -lm -g HW/<file name>; ./a.out
```
Checking for memory leaks with valgrind
```console
valgrind --leak-check=yes ./a.out
```
mpicc multithreading compile
```
mpicc -g -Wall <file name> && ./a.out
```
run with 4 threads
```
mpiexec -n 4 ./a.out
```

#### On cisc372 Server
everything is same as before but,

mpicc multithreading
```
mpicc -g -Wall <file name> && ./a.out
```
run with 4 threads
```
srun -n 4 ./a.out
```