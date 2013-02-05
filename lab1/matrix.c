#include "matrix.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

float *** matrix_a;
float *** matrix_b;
float *** matrix_result;
int i, j, k;

unsigned long long int rdtsc()
{
  unsigned long long int c,d;
  asm volatile("rdtsc" : "=a" (c), "=d" (d));  //assembly code running the instruction
  return c | (d << 32); // calculating the tick value.
}

int main() {
  srand(time(NULL));
  createMatrixes();

  compute();

  freeMatrix(&matrix_a);
  freeMatrix(&matrix_b);
  freeMatrix(&matrix_result);
  return 0;
}

void createMatrixes() {
  matrix_a = (float***) calloc(N, sizeof(float**));
  matrix_b = (float***) calloc(N, sizeof(float**));
  matrix_result = (float***) calloc(N, sizeof(float**));
  for (i = 0; i < N; i++) {
    matrix_a[i] = (float**) calloc(N, sizeof(float*));
    matrix_b[i] = (float**) calloc(N, sizeof(float*));
    matrix_result[i] = (float**) calloc(N, sizeof(float*));
    for (j = 0; j < N; j++) {
      matrix_a[i][j] =  (float*) calloc(K*K, sizeof(float));
      matrix_b[i][j] =  (float*) calloc(K*K, sizeof(float));
      matrix_result[i][j] =  (float*) calloc(K*K, sizeof(float));
      for (k = 0; k < K*K; k++) {
        matrix_a[i][j][k] = (float) rand() / (float) FLOATMAX;
        matrix_b[i][j][k] = (float) rand() / (float) FLOATMAX;
      }
    }
  }
}

void freeMatrix(float **** matrix) {
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
     free((*matrix)[i][j]); 
    }
    free((*matrix)[i]);
  }
  free((*matrix));
  (*matrix) = NULL;
}

void computeMatrix(float * a, float * b, float * result) {
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < K; j++) {
      for (int k = 0; k < K; k++) {
        result[i*4+j] += a[i*4+k]*b[j*4+k];
      }
    }
  }
}

void compute() {
  int i, j;
  unsigned long long time1, time2;
  time1 = rdtsc();
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      computeMatrix(matrix_a[i][j], matrix_b[i][j], matrix_result[i][j]);
   }
  }
  time2 = rdtsc();
  #ifdef BUILD_NAME
    printf(BUILD_NAME);
    printf(": ");
  #endif
  printf("%llu ticks.\n", time2 - time1);
}

