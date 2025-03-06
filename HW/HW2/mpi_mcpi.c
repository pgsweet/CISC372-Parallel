#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
/*** TODO 1: include the required MPI header file ***/

/* struct to store 2D point coordinates 
 */
typedef struct _point {
    double x;
    double y;
} Point;

/* random_point generates coordinates of a random point within the square
 * Output: Point - random point 
 */
Point random_point() {
    Point p;
    p.x = (double)rand()/(double)RAND_MAX;
    p.y = (double)rand()/(double)RAND_MAX;
    return p;
}

/* is_within_circle cheks whether a point falls within the circle 
 * Input: Point - the point to check
 * Output: unsigned short int - 1 if point is within circle, 0 otherwise 
 */
unsigned short int is_within_circle(Point p) {
    double r2 = p.x*p.x + p.y*p.y;
    if (r2 <= 1.0) {
        return 1;
    } else {
        return 0;
    }
}

/* compute_pi calculates the approximate value of pi 
 * using the Monte Carlo method 
 * Input: int - number of points to generate randomly
 * Output: double - the approximate value of pi
 */
double serial_pi(int n) {
    int count=0;
    for (int i=0; i<n; i++) {
        /* generate to random numbers */
        Point p = random_point();
        if (is_within_circle(p)) count++;    
    }
    return  4.0*count/n;
}

double parallel_pi(int n, int comm_sz, int my_rank) {
    int local_n=n, local_count=0, count=0;
    
    /*** TODO 2: add code to determine the number of points that are handled by each process ***/

    for (int i=0; i<local_n; i++) {
        /* generate to random numbers */
        Point p = random_point();
        if (is_within_circle(p))  local_count++;
    }
    
    /*** TODO 3: add code that correctly determines the overall count ***/

    return 4.0*count/n;
}

/* main reads the command line argument and outputs the approximate value of pi 
 */
int main(int argc, char *argv[]) {
    long n;
    double pi, error;
    double start, finish, elapsed_serial, elapsed_parallel;

	/*** TODO 4: add code to set up the MPI execution environment ***/

    int comm_sz=1, my_rank=0;
    /*** TODO 5: add code to set comm_sz and my_rank to the appropriate values ***/
    
    if (my_rank == 0) {
        /* Check command line arguments for correctness and read the parameters from argv */
        if (argc != 2) {
            printf("Usage: pi <npoints>\n");
            exit(-1);
        } else {
            n = atoi(argv[1]);
        }
    }
    /*** TODO 6: Add code that ensures that every process has the input value for n */

    /* set a random seed */
    srand(1);

    /* compute approximate value of pi and calculate difference to real value */
    GET_TIME(start);
    pi = serial_pi(n);
    error = pi - 4.0*atan(1);
    GET_TIME(finish);
    elapsed_serial = finish - start;

	/*** TODO 7: add code to synchronize the processes ***/

    GET_TIME(start);
    pi = parallel_pi(n, comm_sz, my_rank);
    error = pi - 4.0*atan(1);
    GET_TIME(finish);
    elapsed_parallel = finish - start;

    if (my_rank == 0) {
        /* Print the results to stout */
        printf("The approximate value of pi is pi=%f (error=%f)\n", pi, error);

        /* Print performance measurements */
        printf("Serial runtime: %f s\n", elapsed_serial);
        printf("Parallel runtime using %d processes: %f s\n", comm_sz, elapsed_parallel);
        printf("Speedup: %.3f\n", elapsed_serial/elapsed_parallel);
    }

	/*** TODO 8: don't forget to terminate the MPI execution environment */

    return 0;
}

/* you can compile and run the program on cisc372.cis.udel.edu using the following commands:
 *   mpicc mpi_mcpi.c -lm -o pi
 *   srun -n <numprocs> ./pi
 */