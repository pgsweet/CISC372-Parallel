/* File:    trap.c
 * Purpose: Calculate definite integral using trapezoidal 
 *          rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -o trap trap.c -lpthread
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * IPP2:    Section 3.2.1 (pp. 101 and ff.) and 5.2 (p. 228)
 */

#include <stdio.h>
#include <pthread.h>

const int MAX_THREADS = 1024;

int thread_count;
double global_integral = 0.0;
pthread_mutex_t mutex;

double a, b;
int n;
double h;

double f(double x);    /* Function we're integrating */
double Trap(double a, double b, int n, double h);
void* Parallel_Trap(void* rank);
 
int main(int argc, char* argv[]) {
    double  integral;   /* Store result in integral   */
    double  a, b;       /* Left and right endpoints   */
    int     n;          /* Number of trapezoids       */
    double  h;          /* Height of trapezoids       */

    long thread;
    pthread_t* thread_handles;

    if (argc != 2) {
        fprintf(stderr, "usage %s <number of threads>\n", argv[0]);
        return 0;
    }
    thread_count = strtol(argv[1], NULL, 10);
    if (thread_count <= 0 || thread_count > MAX_THREADS) {
        fprintf(stderr, "thread_count must be > 0 and <= %d\n", MAX_THREADS);
        return 0;
    }

    thread_handles = (pthread_t*)malloc(thread_count*sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);
 
    printf("Enter a, b, and n\n");
    scanf("%lf", &a);
    scanf("%lf", &b);
    scanf("%d", &n);
 
    h = (b-a)/n;
    
    for (thread = 0; thread < thread_count; thread++) {
        pthread_create(&thread_handles[thread], NULL, Parallel_Trap, (void*)thread);
    }
 
    for (thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15f\n",
        a, b, global_integral);
 
    return 0;
}  /* main */
 
/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double Trap(double a, double b, int n, double h) {
    double integral;
    int k;
 
    integral = (f(a) + f(b))/2.0;
    for (k = 1; k <= n-1; k++) {
        integral += f(a+k*h);
    }
    integral = integral*h;
 
    return integral;
 }  /* Trap */


 void* Parallel_Trap(void* rank) {
    long my_rank = (long)rank;
    double local_a, local_b;
    int local_n;
    double local_integral;

    local_n = n / thread_count;
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    local_integral = Trap(local_a, local_b, local_n, h);

    pthread_mutex_lock(&mutex);
    global_integral += local_integral;
    pthread_mutex_unlock(&mutex);
 }
 
 /*------------------------------------------------------------------
  * Function:    f
  * Purpose:     Compute value of function to be integrated
  * Input args:  x
  */
 double f(double x) {
    double return_val;
 
    return_val = x*x;
    return return_val;
 }  /* f */
 