#include <stdio.h>

void Print_vector(const char title[], const float x[], const int n) {
   printf("%s = ", title);
   for (int i = 0; i < n; i++)
      printf("%.2f ", x[i]);
   printf("\n");
}  /* Print_vector */

void Rand_vector(float x[], const int n) {
   for (int i = 0; i < n; i++)
      x[i] = random()/((double) RAND_MAX);
}  /* Rand_vector */

float Dot_prod(const float a[], const float b[], const int n) {
    float dp = 0.0;
    for (int i=0; i<n; i++) {
        dp += a[i] + b[i];
    }
    return dp;
} /* Vec_add */

int main(int argc, char *argv[]) {
    int blk_ct, th_per_blk;
    int n;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <n> <blk_ct> <th_per_blk>\n", argv[0]);
        exit(0);
    } else {
        n = strtol(argv[1], NULL, 10);
        blk_ct = strtol(argv[2], NULL, 10);
        th_per_blk = strtol(argv[3], NULL, 10);
    }

    float *a = (float*)malloc(n*sizeof(float));
    float *b = (float*)malloc(n*sizeof(float));
    
    srandom(12345);
    Rand_vector(a, n);
    Rand_vector(b, n);

    float dp = Dot_prod(a, b, n);

    Print_vector("a", a, n);
    Print_vector("b", b, n);
    printf("Dot product = %f\n", dp);

    free(a);
    free(b);

    return 0;
} /* main */