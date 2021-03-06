#include "matrix.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef USE_TCMALLOC
  #include <google/tcmalloc.h>
  #define CALLOC tc_calloc
  #define CFREE  tc_cfree
#else
  #define CALLOC calloc
  #define CFREE  free
#endif

float * matrix_a;
float * matrix_b;
float * matrix_result;
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
  matrix_a = (float*) CALLOC((N*N*K*K), sizeof(float));
  matrix_b = (float*) CALLOC((N*N*K*K), sizeof(float));
  matrix_result = (float*) CALLOC((N*N*K*K), sizeof(float));
  for (int i = 0; i < N*N*K*K; i++) {
    matrix_a[i] = (float)rand() / (float)FLOATMAX;
    matrix_b[i] = (float)rand() / (float)FLOATMAX;
  }
}

void freeMatrix(float ** matrix) {
  CFREE((*matrix));
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
  int i;
  unsigned long long time1, time2;
  time1 = rdtsc();
  #ifdef _OPENMP
    printf("OpenMP: using %d parallel threads...\n", omp_get_max_threads());
    #pragma omp parallel for
  #endif
  for (i = 0; i < N*N; i++) {
    computeMatrix(&matrix_a[i*K*K], &matrix_b[i*K*K], &matrix_result[i*K*K]);
  }
  time2 = rdtsc();
  printf("%llu ticks", time2 - time1);
  #ifdef BUILD_NAME
    printf(": ");
    printf(BUILD_NAME);
  #endif
  printf("\n");
}


