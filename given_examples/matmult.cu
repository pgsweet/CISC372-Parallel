#include <stdio.h>

#define BLOCK_SIZE 16

typedef struct {
    int height;
    int width;
    float *elements;
} Matrix;

void PrintMat(const char title[], const Matrix M) {
    printf("%s = ", title);
    for (int i = 0; i < M.height; i++) {
        for (int j = 0; j < M.width; j++) {
            printf("%.2f ", M.elements[i*M.width+j]);
        }
        printf("\n");
    }
    printf("\n");
}  /* PrintMat */

void RandMat(const Matrix M) {
    int m = M.height;
    int n = M.width;
    for (int i=0; i<m*n; i++) {
        M.elements[i] = random()/((double) RAND_MAX);
    }
} /* RandMat */

void MatMulSerial(const Matrix A, const Matrix B, Matrix C) {
    for (int i=0; i<C.height; i++) {
        for (int j=0; j<C.width; j++) {
            C.elements[i*C.width+j] = 0.0;
            for (int k=0; k<A.width; k++) {
                C.elements[i*C.width+j] += A.elements[i*A.width+k]*B.elements[k*B.width+j];
            }
        }
    }
} /* MatMulSerial */

int main(int argc, char* argv[]) {
    int m, k, n;
    
    if (argc != 4) {
        fprintf(stderr, "usage: %s <m> <k> <n>\n", argv[0]);
        exit(0);
    } else {
        m = strtol(argv[1], NULL, 10);
        k = strtol(argv[2], NULL, 10);
        n = strtol(argv[3], NULL, 10);
    }

    Matrix A = { m, k, NULL };
    Matrix B = { k, n, NULL };
    Matrix C = { m, n, NULL };

    A.elements = (float *)malloc(m*k*sizeof(float));
    B.elements = (float *)malloc(k*n*sizeof(float));
    C.elements = (float *)malloc(m*n*sizeof(float));

    srandom(12345);
    RandMat(A);
    RandMat(B);

    MatMulSerial(A, B, C);

    PrintMat("C", C);

    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
} /* main */