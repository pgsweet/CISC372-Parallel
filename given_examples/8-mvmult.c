#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* function to print matrix */
void print_vector(int M, int *y) {
    /* print result vector y */
    if (M <= 6) {
        printf("y = < %d, ",y[0]);
        for (int i=1; i<M-1; i++) {
            printf("%d, ", y[i]);
        }
        printf("%d >\n",y[M-1]);
    } else {
        printf("y = < %d, ",y[0]);
        for (int i=1; i<3; i++) {
            printf("%d, ", y[i]);
        }
        printf("..., ");
        for (int i=M-3; i<M-1; i++) {
            printf("%d, ", y[i]);
        }
        printf("%d >\n",y[M-1]);
    }
}

/* function to initialize matrix */
void init_matrix(int M, int N, int **A, int *x) {
    /* initialize matrix A and vector x */
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            A[i][j] = 1 + j + i*N;
            x[j] = pow(-1,j);
        }
    }
}

/* function to perform matrix multiplication y=A*x */
void matmult(int M, int N, int **A, int *x, int *y) {

    /* matrix vector multiplication y = A*x */
    for (int i=0; i<M; i++) {
        y[i] = 0;
        for (int j=0; j<N; j++) {
            y[i] += A[i][j] * x[j];
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

    /* TODO: declare and allocate arrays */
    int **A = malloc(M * sizeof(int *));
    for (int m = 0; m < M; m++) {
        A[m] = (int *)malloc(N * sizeof(int));
    }
    int *x = (int *)malloc(N * sizeof(int));
    int *y = (int *)malloc(M * sizeof(int));

    init_matrix(M, N, A, x);

    matmult(M, N, A, x, y);

    print_vector(M, y);

    /* TODO: free arrays */
    for (int m = 0; m < M; m++) {
        free(A[m]);
    }
    free(A);
    free(x);
    free(y);

    return 0;
}