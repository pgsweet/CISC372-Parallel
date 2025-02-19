#include <stdio.h>
#include <stdlib.h>

#define M 5
#define N 6

void matrixArrayOfArrays(int m, int n) {
    // Part A
    float** matrix = (float**)malloc(m*sizeof(float*));
    for (int i = 0; i < m; i++) {
        matrix[i] = (float*)malloc(n*sizeof(float));
    }
    // Part B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (float)(i*n) + j + 1;
        }
    }
    // Part C
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
    // Part D
    printf("\n");
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
    // Part E
    for (int i = 0; i < m; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


int main(int argc,char** argv){
    matrixArrayOfArrays(M,N);
    // matrixOneBigArray(M,N);

    return 0;
}