#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

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

/* main reads the command line argument and outputs the approximate value of pi 
 */
int main(int argc, char *argv[]) {
    int n;
    double pi, error;
    double start, finish, elapsed;

    /* Check command line arguments for correctness and read the parameters from argv */
    if (argc != 2) {
        printf("Usage: pi <npoints>\n");
        exit(-1);
    } else {
        n = atoi(argv[1]);
    }

    /* set a random seed */
    srand(1);

    /* compute approximate value of pi and calculate difference to real value */
    GET_TIME(start);
    pi = serial_pi(n);
    error = pi - 4.0*atan(1);
    GET_TIME(finish);
    elapsed = finish - start;

    /* Print the results to stout */
    printf("The approximate value of pi is pi=%f (error=%f)\n", pi, error);

    /* Print performance measurements */
   	printf("Serial runtime: %f s\n", elapsed);

    return 0;
}