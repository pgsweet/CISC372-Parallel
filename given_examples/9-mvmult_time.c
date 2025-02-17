#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

/* function to perform matrix multiplication y=A*x */
void matmult(int m, int n, int *Ap, int *x, int *y) {
    int (*A)[n] = (int (*)[])Ap;

    for (int i=0; i<m; i++) {
        y[i] = 0;
        for (int j=0; j<n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }

}

/* function to perform matrix multiplication y=B^t*x */
void matmult2(int m, int n, int *Bp, int *x, int *y) {
    int (*B)[m] = (int (*)[])Bp;

    for (int i=0; i<m; i++) {
        y[i] = 0;
        for (int j=0; j<n; j++) {
            y[i] += x[j] * B[j][i];
        }
    }

}

int main(int argc, char *argv[]) {
    double start, finish, elapsed;

    /* quick check that we got two arguments from the command line */
    if (argc<=2) {
        fprintf(stderr, "Usage: a.out <M> <N>\n");
        return -1;
    }

    /* read array dimensions from command line arguments */
    unsigned int M = atoi(argv[1]);
    unsigned int N = atoi(argv[2]);

    /* declare and allocate arrays */
    int *x = malloc(N*sizeof(int));
    int *y = malloc(M*sizeof(int));

    /* allocate MxN matrix A */
    int *A = malloc(M*N*sizeof(int));

    /* initialize matrix A and vector x */
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            A[i*N+j] = 1 + j + i*N;
            x[j] = pow(-1,j);
        }
    }

    /* allocate NxM matrix B */
    int *B = malloc(M*N*sizeof(int));

    /* inititalize matrix B = A^t */
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            B[j*M+i] = A[i*N+j];
        }
    }
    
    GET_TIME(start);
    matmult(M, N, A, x, y);
    GET_TIME(finish);
    elapsed = finish - start;

    printf("The matrix-vector multiplication took %f seconds.\n", elapsed);

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

    GET_TIME(start);
    matmult2(M, N, B, x, y);
    GET_TIME(finish);
    elapsed = finish - start;

    printf("The transposed matrix-vector multiplication took %f seconds.\n", elapsed);

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

    free(x);
    free(y);
    free(A);
    free(B);

}