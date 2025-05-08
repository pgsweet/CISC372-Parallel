#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

/* function to perform matrix multiplication y=A*x */
void matmult(int m, int n, int *A, int *x, int *y) {

    for (int i=0; i<m; i++) {
        y[i] = 0;
        for (int j=0; j<n; j++) {
            y[i] += A[i*n+j] * x[j];
        }
    }

}

__global__ void matmult_kernel(int m, int n, int *A, int *x, int *y) {

    for (int i=0; i<m; i++) {
        y[i] = 0;
        for (int j=0; j<n; j++) {
            y[i] += A[i*n+j] * x[j];
        }
    }

}

int main(int argc, char *argv[]) {
    //double start, finish, elapsed;

    /* quick check that we got two arguments from the command line */
    if (argc<=2) {
        fprintf(stderr, "Usage: a.out <M> <N>\n");
        return -1;
    }

    /* read array dimensions from command line arguments */
    unsigned int M = atoi(argv[1]);
    unsigned int N = atoi(argv[2]);

    /* declare and allocate arrays */
    int *x, *y, *A;
    cudaMallocManaged(&y, M*sizeof(int));
    cudaMallocManaged(&x, N*sizeof(int));
    cudaMallocManaged(&A, M*N*sizeof(int));

    /* initialize matrix A and vector x */
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            A[i*N+j] = 1 + j + i*N;
            x[j] = pow(-1,j);
        }
    }

    matmult_kernel <<<1, 1>>> (M, N, A, x, y);
    cudaDeviceSynchronize();
    
    //printf("The matrix-vector multiplication took %f seconds.\n", elapsed);

    // /* print result vector y */
    // if (M <= 6) {
    //     printf("y = < %d, ",y[0]);
    //     for (int i=1; i<M-1; i++) {
    //         printf("%d, ", y[i]);
    //     }
    //     printf("%d >\n",y[M-1]);
    // } else {
    //     printf("y = < %d, ",y[0]);
    //     for (int i=1; i<3; i++) {
    //         printf("%d, ", y[i]);
    //     }
    //     printf("..., ");
    //     for (int i=M-3; i<M-1; i++) {
    //         printf("%d, ", y[i]);
    //     }
    //     printf("%d >\n",y[M-1]);
    // }

    int *y_ser = (int *)malloc(M*sizeof(int));
    matmult(M, N, A, x, y_ser);
    float maxError = 0.0;
    for (int i=0; i<M; i++) maxError = fmax(maxError, fabs(y[i]-y_ser[i]));
    printf("Max error: %f\n", maxError);
    free(y_ser);

    cudaFree(x);
    cudaFree(y);
    cudaFree(A);

}