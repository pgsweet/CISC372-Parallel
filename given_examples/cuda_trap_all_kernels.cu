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
#include "timer.h"

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
__host__ __device__ float f(const float x) {
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
   float h = (b-a)/n;
   float trap = 0.5*(f(a) + f(b));

   for (int i = 1; i <= n-1; i++) {
      trap += f(a + i*h);
   }
   trap = trap*h;

   return trap;
}  /* Trap */

#ifdef REDUCE_ATOMIC
__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *trap) {
   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   float my_val = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) my_val += f(a+i*h);
   atomicAdd(trap, my_val);
}
#endif

#ifdef REDUCE_GLOBAL
__device__ float Tree_Sum(float *values, const int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   for (int offset=1; offset<blockDim.x; offset<<=1) {
      int source = i+offset;
      if (i%(offset<<1) == 0 && source < n) values[i] += values[source];
      __syncthreads();
   }
   return values[i];
}

__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *trap, float *values) {
   int gidx = blockIdx.x * blockDim.x + threadIdx.x;

   values[gidx] = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) values[gidx] += f(a+i*h);
   __syncthreads();

   float blk_sum = Tree_Sum(values, n);

   if (threadIdx.x == 0) atomicAdd(trap, blk_sum);
}
#endif

#ifdef REDUCE_SHARED
__device__ float Tree_Sum(float *values) {
   int i = threadIdx.x;
   for (int offset=1; offset<blockDim.x; offset<<=1) {
      int source = i+offset;
      if (i%(offset<<1) == 0) values[i] += values[source];
      __syncthreads();
   }
   return values[i];
}

__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *trap) {
   __shared__ float values[1024];

   int gidx = blockIdx.x * blockDim.x + threadIdx.x;

   values[threadIdx.x] = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) values[threadIdx.x] += f(a+i*h);
   __syncthreads();

   float blk_sum = Tree_Sum(values);

   if (threadIdx.x == 0) atomicAdd(trap, blk_sum);
}
#endif

#ifdef REDUCE_WARP
__device__ float Warp_Sum(float* values) {
   int lane = threadIdx.x % warpSize;
   for (int offset=1; offset<warpSize; offset<<=1) {
      int source = (lane+offset) % warpSize;
      values[lane] += values[source];
   }
   return values[lane];
}

__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *trap) {
   __shared__ float values[1024];
   __shared__ float warp_sums[32];

   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   int warp = threadIdx.x / warpSize;
   int lane = threadIdx.x % warpSize;

   float warp_sum, blk_sum;

   values[threadIdx.x] = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) values[threadIdx.x] += f(a+i*h);

   warp_sum = Warp_Sum(values+warp*warpSize);
   if (lane == 0) warp_sums[warp] = warp_sum;
   __syncthreads();

   if (lane >= (blockDim.x + warpSize - 1)/warpSize) warp_sums[lane] = 0.0f;
   if (warp == 0) blk_sum = Warp_Sum(warp_sums);

   if (threadIdx.x == 0) atomicAdd(trap, blk_sum);

}  /* Trap_Kernel */
#endif

#ifdef REDUCE_SHFL
__device__ float Warp_Sum(float val) {
   unsigned mask = 0xffffffff;
   for (int stride=1; stride<warpSize; stride<<=1) {
      val += __shfl_down_sync(mask, val, stride);
   }
   return val;
}

__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *trap) {
   __shared__ float warp_sums[32];
   float value, warp_sum, blk_sum;

   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   int warp = threadIdx.x / warpSize;
   int lane = threadIdx.x % warpSize;

   value = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) value += f(a+i*h);

   warp_sum = Warp_Sum(value);
   if (lane == 0) warp_sums[warp] = warp_sum;
   __syncthreads();

   if (warp == 0) blk_sum = Warp_Sum((lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane] : 0.0f);

   if (threadIdx.x == 0) atomicAdd(trap, blk_sum);

}  /* Trap_Kernel */
#endif

#ifdef REDUCE_LASTBLOCK
__device__ int counter = 0;

__device__ bool lastBlock(int* counter) {
    __threadfence(); //ensure that partial result is visible by all blocks
    int last = 0;
    if (threadIdx.x == 0)
        last = atomicAdd(counter, 1);
    return __syncthreads_or(last == gridDim.x-1);
}

__device__ float Warp_Sum(float val) {
   unsigned mask = 0xffffffff;
   for (int stride=1; stride<warpSize; stride<<=1) {
      val += __shfl_down_sync(mask, val, stride);
   }
   return val;
}

__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *trap, float *blk_sums) {
   __shared__ float warp_sums[32];

   float my_val, warp_sum, blk_sum;
   
   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   int warp = threadIdx.x / warpSize;
   int lane = threadIdx.x % warpSize;
   
   my_val = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) my_val += f(a+i*h);

   warp_sum = Warp_Sum(my_val);
   if (lane == 0) warp_sums[warp] = warp_sum;
   __syncthreads();

   if (warp == 0) blk_sum = Warp_Sum((lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane] : 0.0f);

   if (threadIdx.x == 0) blk_sums[blockIdx.x] = blk_sum;

   if (lastBlock(&counter)) {
      my_val = 0.0;
      for (int i=threadIdx.x; i<gridDim.x; i+=blockDim.x) my_val += blk_sums[i];
      warp_sum = Warp_Sum(my_val);
      if (lane == 0) warp_sums[warp] = warp_sum;
      __syncthreads();
      if (warp == 0) blk_sum = Warp_Sum((lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane] : 0.0f);
      if (threadIdx.x == 0) *trap = blk_sum;
   }
}  /* Trap_Kernel */
#endif

#ifdef REDUCE_MULTBLOCK
__device__ float Warp_Sum(float value) {
   unsigned mask = 0xffffffff;
   for (int offset=1; offset<warpSize; offset<<=1) {
      value += __shfl_down_sync(mask, value, offset);
   }
   return value;
}

__global__ void Sum_Array_Kernel(const float *values, const int n, float *blk_sums) {
   __shared__ float warp_sums[32];
   float my_val, warp_sum, blk_sum;

   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   int warp = threadIdx.x / warpSize;
   int lane = threadIdx.x % warpSize;

   my_val = 0.0f;
   for (int i=gidx; i<n; i+=gridDim.x*blockDim.x) my_val += values[i];

   warp_sum = Warp_Sum(my_val);
   if (lane == 0) warp_sums[warp] = warp_sum;
   __syncthreads();

   if (warp == 0) blk_sum = Warp_Sum((lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane] : 0.0f);

   if (threadIdx.x == 0) blk_sums[blockIdx.x] = blk_sum;
}

/* Trap_Kernel */
__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *blk_sums) {
   __shared__ float warp_sums[32];
   float my_val, warp_sum, blk_sum;

   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   int warp = threadIdx.x / warpSize;
   int lane = threadIdx.x % warpSize;

   my_val = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) my_val += f(a+i*h);

   warp_sum = Warp_Sum(my_val);
   if (lane == 0) warp_sums[warp] = warp_sum;
   __syncthreads();

   if (warp == 0) blk_sum = Warp_Sum((lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane] : 0.0f);

   if (threadIdx.x == 0) blk_sums[blockIdx.x] = blk_sum;

}  /* Trap_Kernel */
#endif

#if (!defined REDUCE_ATOMIC && !defined REDUCE_GLOBAL && !defined REDUCE_SHARED && !defined REDUCE_WARP && !defined REDUCE_SHFL && !defined REDUCE_MULTBLOCK && !defined REDUCE_LASTBLOCK)
__device__ float Warp_Sum(float value) {
   unsigned mask = 0xffffffff;
   for (int offset=1; offset<warpSize; offset<<=1) {
      value += __shfl_down_sync(mask, value, offset);
   }
   return value;
}

__global__ void Sum_Array_Kernel(const float *values, const int n, float *blk_sums) {
   __shared__ float warp_sums[32];
   float my_val, warp_sum, blk_sum;

   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   int warp = threadIdx.x / warpSize;
   int lane = threadIdx.x % warpSize;

   my_val = 0.0f;
   for (int i=gidx; i<n; i+=gridDim.x*blockDim.x) my_val += values[i];

   warp_sum = Warp_Sum(my_val);
   if (lane == 0) warp_sums[warp] = warp_sum;
   __syncthreads();

   if (warp == 0) blk_sum = Warp_Sum((lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane] : 0.0f);

   if (threadIdx.x == 0) blk_sums[blockIdx.x] = blk_sum;
}

/* Trap_Kernel */
__global__ void Trap_Kernel(const float a, const float b, const int n, const float h, float *trap) {
   __shared__ float warp_sums[32];
   float my_val, warp_sum, blk_sum;

   int gidx = blockIdx.x * blockDim.x + threadIdx.x;
   int warp = threadIdx.x / warpSize;
   int lane = threadIdx.x % warpSize;

   my_val = 0.0f;
   for (int i=(0 < gidx) ? gidx : gridDim.x*blockDim.x; i<n; i+=gridDim.x*blockDim.x) my_val += f(a+i*h);

   warp_sum = Warp_Sum(my_val);
   if (lane == 0) warp_sums[warp] = warp_sum;
   __syncthreads();

   if (warp == 0) blk_sum = Warp_Sum((lane < (blockDim.x + warpSize - 1)/warpSize) ? warp_sums[lane] : 0.0f);

   if (threadIdx.x == 0) atomicAdd(trap, blk_sum);
}  /* Trap_Kernel */
#endif

__host__ void Trap(const float a, const float b, const int n, float *trap) {
   float h;

   h = (b-a)/n;

   *trap = 0.5*(f(a)+f(b));

   int th_per_blk = 1024;
   int blk_ct = (n + th_per_blk - 1)/th_per_blk;

#ifdef REDUCE_ATOMIC
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, trap);
#elif defined REDUCE_GLOBAL
   float *values;
   cudaMalloc(&values, n*sizeof(float));
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, trap, values);
   cudaFree(values);
#elif defined REDUCE_SHARED
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, trap);
#elif defined REDUCE_WARP
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, trap);
#elif defined REDUCE_SHFL
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, trap);
#elif defined REDUCE_LASTBLOCK
   float *blk_sums;
   cudaMalloc(&blk_sums, blk_ct*sizeof(float));
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, trap, blk_sums);
   cudaFree(blk_sums);
#elif defined REDUCE_MULTBLOCK
   float *blk_sums;
   cudaMalloc(&blk_sums, blk_ct*sizeof(float));
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, blk_sums);
   Sum_Array_Kernel <<<1, th_per_blk>>> (blk_sums, blk_ct, trap);
   cudaFree(blk_sums);
#else
   Trap_Kernel <<<blk_ct, th_per_blk>>> (a, b, n, h, trap);
#endif
   cudaDeviceSynchronize();

   *trap = *trap*h;
}

int main(int argc, char *argv[]) {
   double start, finish, elapsed;

   float  a=1, b=3; /* Left and right endpoints   */
   int    n;        /* Number of trapezoids       */

   if (argc != 2) {
        fprintf(stderr, "usage: %s <n>\n", argv[0]);
        exit(EXIT_FAILURE);
   } else {
        n = strtol(argv[1], NULL, 10);
   }

   float serial_integral, *cuda_integral;
   cudaMallocManaged(&cuda_integral, sizeof(float));

   GET_TIME(start);
   serial_integral = Serial_Trap(a, b, n);
   GET_TIME(finish);
   elapsed = finish-start; if(elapsed);

   printf("Serial: With n = %d trapezoids, our estimate ", n);
   printf("of the integral from %f to %f = %f\n", a, b, serial_integral);
   fflush(stdout);

   GET_TIME(start);
   Trap(a, b, n, cuda_integral);
   GET_TIME(finish);

   printf("CUDA: With n = %d trapezoids, our estimate ", n);
   printf("of the integral from %f to %f = %f\n", a, b, *cuda_integral);

   printf("Error = %f\n", serial_integral-(*cuda_integral));

#ifdef PERF
   printf("\nSerial runtime: %f s\n", elapsed);
   printf("CUDA runtime: %f s\n", finish-start);
   printf("Speedup = %.2f\n", elapsed/(finish-start));
#endif

   cudaFree(cuda_integral);

   return 0;
}  /* main */