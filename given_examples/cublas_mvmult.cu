#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include "timer.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/* function to perform matrix multiplication y=A*x */
void matmult(int m, int n, double *A, double *x, double *y) {

    for (int i=0; i<m; i++) {
        y[i] = 0;
        for (int j=0; j<n; j++) {
            y[i] += A[i*n+j] * x[j];
        }
    }

}

__device__ float Tree_Sum(int *values) {
    int i = threadIdx.x;
    for (int stride=1; stride<blockDim.x; stride<<=1) {
        int idx = i*stride*2;
        values[idx] += idx+stride<blockDim.x ? values[idx+stride] : 0;
        __syncthreads();
    }
    return values[i];
}

__global__ void matmult_kernel(int m, int n, double *A, double *x, double *y) {
    __shared__ int shared_values[1024];

    int row = blockIdx.x;
    int i = threadIdx.x;

    shared_values[i] = i<n ? A[row*n+i]*x[i] : 0;
    __syncthreads();

    float blk_sum = Tree_Sum(shared_values);
    if (threadIdx.x == 0) y[row] = blk_sum;

}

int main(int argc, char *argv[]) {
    double start, finish, elapsed;

    /* quick check that we got two arguments from the command line */
    if (argc<=2) {
        fprintf(stderr, "Usage: a.out <M> <N>\n");
        return -1;
    }

    /* read array dimensions from command line arguments */
    unsigned long M = atoi(argv[1]);
    unsigned long N = atoi(argv[2]);

    /* declare and allocate arrays */
    double *x, *y, *A;
    // cudaMallocManaged(&y, M*sizeof(double));
    // cudaMallocManaged(&x, N*sizeof(double));
    // cudaMallocManaged(&A, M*N*sizeof(double));
    A = (double *)malloc(M*N*sizeof(double));
    x = (double *)malloc(N*sizeof(double));
    y = (double *)malloc(M*sizeof(double));

    /* initialize matrix A and vector x */
    for (long i=0; i<M; i++) {
        for (long j=0; j<N; j++) {
            A[i*N+j] = 1 + j + i*N;
            x[j] = pow(-1,j);
            y[i] = 0.0;
        }
    }

    double *y_ser = (double *)malloc(M*sizeof(double));   
    GET_TIME(start);
    matmult(M, N, A, x, y_ser);
    GET_TIME(finish);
    elapsed = finish - start; if(elapsed);

    double *dev_x, *dev_y, *dev_A;
    HANDLE_ERROR( cudaMalloc(&dev_y, M*sizeof(double)) );
    HANDLE_ERROR( cudaMalloc(&dev_x, N*sizeof(double)) );
    HANDLE_ERROR( cudaMalloc(&dev_A, M*N*sizeof(double)) );

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0;
    const double beta = 0.0;

    HANDLE_ERROR( cudaMemcpy(dev_A, A, N*M*sizeof(double), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_x, x, N*sizeof(double), cudaMemcpyHostToDevice) );

    GET_TIME(start);
    cublasDgemv(handle, CUBLAS_OP_T, N, M, &alpha, dev_A, N, dev_x, 1, &beta, dev_y, 1);
    cudaDeviceSynchronize();
    GET_TIME(finish);

    HANDLE_ERROR( cudaMemcpy(y, dev_y, M*sizeof(double), cudaMemcpyDeviceToHost) );

    cublasDestroy(handle);

    /* print result vector y */
    if (M <= 6) {
        printf("y = < %.0f, ",y[0]);
        for (int i=1; i<M-1; i++) {
            printf("%.0f, ", y[i]);
        }
        printf("%.0f >\n",y[M-1]);
    } else {
        printf("y = < %.0f, ",y[0]);
        for (int i=1; i<3; i++) {
            printf("%.0f, ", y[i]);
        }
        printf("..., ");
        for (int i=M-3; i<M-1; i++) {
            printf("%.0f, ", y[i]);
        }
        printf("%.0f >\n",y[M-1]);
    }

    double maxError = 0.0;
    for (int i=0; i<M; i++) maxError = fmax(maxError, fabs(y[i]-y_ser[i]));
    printf("Max error: %f\n", maxError);

#ifdef PERF
   printf("\nSerial runtime: %f s\n", elapsed);
   printf("CUDA runtime: %f s\n", finish-start);
   printf("Speedup = %.2f\n", elapsed/(finish-start));
#endif

    free(y_ser);

    HANDLE_ERROR( cudaFree(dev_x) );
    HANDLE_ERROR( cudaFree(dev_y) );
    HANDLE_ERROR( cudaFree(dev_A) );

    // cudaFree(x);
    // cudaFree(y);
    // cudaFree(A);
    free(A);
    free(x);
    free(y);
}

    // cublasHandle_t handle;
    // cublasCreate(&handle);

    // float alpha = 1.0;
    // float beta = 0.0;

    // cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, A, N, x, 1, &beta, y, 1);
    // cudaDeviceSynchronize();