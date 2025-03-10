#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "timer.h"

void get_input(int argc, char *argv[], int *m, int *n) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank == 0) {
        /* quick check that we got two arguments from the command line */
        if (argc<=2) {
            fprintf(stderr, "Usage: a.out <M> <N>\n");
            exit(-1);
        }

        /* read array dimensions from command line arguments */
        *m = atoi(argv[1]);
        *n = atoi(argv[2]);
    }
    MPI_Bcast(m, 1, MPI_INT, 0, comm);
    MPI_Bcast(n, 1, MPI_INT, 0, comm);
} /* get_input */

void init_matrix_vector(int m, int n, int *Ap, int *x) {
    int (*A)[n] = (int (*)[])Ap;
    /* initialize matrix A and vector x */
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            A[i][j]= 1 + j + i*n;
            x[j] = pow(-1,j);
        }
    }
} /* init_matrix_vector */

void print_vector(int m, int *y) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank == 0) {
        /* print result vector y */
        if (m <= 6) {
            printf("y = < %d, ",y[0]);
            for (int i=1; i<m-1; i++) {
                printf("%d, ", y[i]);
            }
            printf("%d >\n",y[m-1]);
        } else {
            printf("y = < %d, ",y[0]);
            for (int i=1; i<3; i++) {
                printf("%d, ", y[i]);
            }
            printf("..., ");
            for (int i=m-3; i<m-1; i++) {
                printf("%d, ", y[i]);
            }
            printf("%d >\n",y[m-1]);
        }
    }
} /* print_vector */

void mvmult(int m, int n, int *Ap, int *x, int *y) {
    int (*A)[n] = (int (*)[])Ap;
    /* perform matrix multiplication y=A*x */
    for (int i=0; i<m; i++) {
        y[i] = 0;
        for (int j=0; j<n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
} /* mvmult */

int main(int argc, char *argv[]) {
    double start, finish, elapsed_serial=0, elapsed_parallel=0, local_elapsed;
    int m, n;
    int *A, *x, *y;

    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_sz, my_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    get_input(argc, argv, &m, &n);

    /* declare and allocate arrays */
    x = malloc(n*sizeof(int));
    y = malloc(m*sizeof(int));
    A = malloc(m*n*sizeof(int));

    init_matrix_vector(m, n, A, x);

    /* SERIAL COMPUTATION */
    
    mvmult(m, n, A, x, y);
   
    print_vector(m, y);

    if (my_rank == 0)
         printf("Serial matrix-vector multiplication took %f s\n", elapsed_serial);

    for (int i=0; i<m; i++) y[i] = 0; // reset output array for good measure

    /* PARALLEL COMPUTATION */

    int local_m, *local_A, *local_y;
    local_m = m/comm_sz;
    local_A = malloc(local_m*n*sizeof(int));
    local_y = malloc(local_m*sizeof(int));

    MPI_Scatter(A, local_m*n, MPI_INT, local_A, local_m*n, MPI_INT, 0, comm);

    mvmult(local_m, n, local_A, x, local_y);

    MPI_Gather(local_y, local_m, MPI_INT, y, local_m, MPI_INT, 0, comm);

    print_vector(m, y);

    if (my_rank == 0) {
         printf("MPI matrix-vector multiplication took %f s using %d processes\n", elapsed_parallel, comm_sz);
         printf("Speedup: %.2f\n", elapsed_serial/elapsed_parallel);
    }

    free(local_y);
    free(local_A);

    free(x);
    free(y);
    free(A);

    MPI_Finalize();

}