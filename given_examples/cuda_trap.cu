/* File:    trap.c
 * Purpose: Calculate definite integral using trapezoidal 
 *          rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -o trap trap.c
 * Usage:   ./trap
 *
 * Notes:    
 *    1.  The function f(x) is hardwired.
 *    2.  This is very similar to the trap.c program
 *        in Chapters 3 and 5,   However, it uses floats
 *        instead of floats.
 *
 * IPP2:  6.10.1 (pp. 314 and ff.)
 */

#include <stdio.h>

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
float f(const float x) {
   float return_val;

   return_val = x*x;
   return return_val;
}  /* f */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n
 * Return val:  Estimate of the integral 
 */
float Serial_Trap(const float a, const float b, const int n) {
   float x, h = (b-a)/n;
   float trap = 0.5*(f(a) + f(b));

   for (int i = 1; i <= n-1; i++) {
      x = a + i*h;
      trap += f(x);
   }
   trap = trap*h;

   return trap;
}  /* Trap */

float Trap(const float a, const float b, const int n) {
   float x, h = (b-a)/n;
   float trap = 0.5*(f(a) + f(b));

   for (int i = 1; i <= n-1; i++) {
      x = a + i*h;
      trap += f(x);
   }
   trap = trap*h;

   return trap;
}  /* Trap */

int main(int argc, char *argv[]) {
   float  a, b;       /* Left and right endpoints   */
   int    n;          /* Number of trapezoids       */

   a = 1;
   b = 3;

   if (argc != 2) {
        fprintf(stderr, "usage: %s <n>\n", argv[0]);
        exit(0);
   } else {
        n = strtol(argv[1], NULL, 10);
   }

   float serial_integral = Serial_Trap(a, b, n);
   printf("Serial: With n = %d trapezoids, our estimate ", n);
   printf("of the integral from %f to %f = %f\n", a, b, serial_integral);

   float cuda_integral = Trap(a, b, n);
   
   printf("CUDA: With n = %d trapezoids, our estimate ", n);
   printf("of the integral from %f to %f = %f\n", a, b, cuda_integral);

   printf("Error = %f\n", serial_integral-cuda_integral);

   return 0;
}  /* main */