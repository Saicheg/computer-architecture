#include "matrix.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <malloc.h>

#include <emmintrin.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

float * matrix_a;
float * matrix_b;
float * matrix_result_auto;
float * matrix_result_manual;
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

  #ifdef _OPENMP
    printf("OpenMP: using up to %d parallel threads...\n", omp_get_max_threads());
  #endif

  compute();

  freeMatrix(&matrix_a);
  freeMatrix(&matrix_b);
  freeMatrix(&matrix_result_auto);
  freeMatrix(&matrix_result_manual);
  return 0;
}

void createMatrixes() {
  matrix_a = (float*) memalign(16, sizeof(float)*(N*N*K*K));
  matrix_b = (float*) memalign(16, sizeof(float)*(N*N*K*K));
  matrix_result_auto = (float*) memalign(16, sizeof(float)*(N*N*K*K));
  matrix_result_manual = (float*) memalign(16, sizeof(float)*(N*N*K*K));
  
  for (int i = 0; i < N*N*K*K; i++) {
    matrix_a[i] = (float)rand() / (float)FLOATMAX;
    matrix_b[i] = (float)rand() / (float)FLOATMAX;
  }
}

void freeMatrix(float ** matrix) {
  free(*matrix);
  (*matrix) = NULL;
}

void computeMatrixAuto(float * a, float * b, float * result) {
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < K; j++) {
      for (int k = 0; k < K; k++) {
        result[i*4+j] += a[i*4+k] * b[k*4+j];
      }
    }
  }
}

void computeMatrixManual(float * a, float * b, float * result) {
  __m128 ar, b1, b2, b3, b4, r1, r2, r3, r4;

  b1 = _mm_load_ps(&b[0*4]);
  b2 = _mm_load_ps(&b[1*4]);
  b3 = _mm_load_ps(&b[2*4]);
  b4 = _mm_load_ps(&b[3*4]);

  ar = _mm_set1_ps(a[0]);
  r1 = _mm_mul_ps(ar, b1);

  ar = _mm_set1_ps(a[1]);
  r1 = _mm_add_ps(r1, _mm_mul_ps(ar, b2));

  ar = _mm_set1_ps(a[2]);
  r1 = _mm_add_ps(r1, _mm_mul_ps(ar, b3));

  ar = _mm_set1_ps(a[3]);
  r1 = _mm_add_ps(r1, _mm_mul_ps(ar, b4));


  ar = _mm_set1_ps(a[4]);
  r2 = _mm_mul_ps(ar, b1);

  ar = _mm_set1_ps(a[5]);
  r2 = _mm_add_ps(r2, _mm_mul_ps(ar, b2));

  ar = _mm_set1_ps(a[6]);
  r2 = _mm_add_ps(r2, _mm_mul_ps(ar, b3));

  ar = _mm_set1_ps(a[7]);
  r2 = _mm_add_ps(r2, _mm_mul_ps(ar, b4));


  ar = _mm_set1_ps(a[8]);
  r3 = _mm_mul_ps(ar, b1);

  ar = _mm_set1_ps(a[9]);
  r3 = _mm_add_ps(r3, _mm_mul_ps(ar, b2));

  ar = _mm_set1_ps(a[10]);
  r3 = _mm_add_ps(r3, _mm_mul_ps(ar, b3));

  ar = _mm_set1_ps(a[11]);
  r3 = _mm_add_ps(r3, _mm_mul_ps(ar, b4));


  ar = _mm_set1_ps(a[12]);
  r4 = _mm_mul_ps(ar, b1);

  ar = _mm_set1_ps(a[13]);
  r4 = _mm_add_ps(r4, _mm_mul_ps(ar, b2));

  ar = _mm_set1_ps(a[14]);
  r4 = _mm_add_ps(r4, _mm_mul_ps(ar, b3));

  ar = _mm_set1_ps(a[15]);
  r4 = _mm_add_ps(r4, _mm_mul_ps(ar, b4));

  _mm_store_ps(&result[0], r1);
  _mm_store_ps(&result[4], r2);
  _mm_store_ps(&result[8], r3);
  _mm_store_ps(&result[12], r4);
}

void compute() {
  unsigned long long time_auto_start, time_auto_end,
                     time_manual_start, time_manual_end;

  time_auto_start = rdtsc();
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (int i = 0; i < N*N; i++) {
    computeMatrixAuto(&matrix_a[i*K*K],
                      &matrix_b[i*K*K],
                      &matrix_result_auto[i*K*K]);
  }
  time_auto_end = rdtsc();

  time_manual_start = rdtsc();
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (int i = 0; i < N*N; i++) {
    computeMatrixManual(&matrix_a[i*K*K],
                        &matrix_b[i*K*K],
                        &matrix_result_manual[i*K*K]);
  }
  time_manual_end = rdtsc();

  int error_count = 0;
  for (long i = 0; i < N*N*K*K; i++) {
    if (fabsf(1 - matrix_result_auto[i]/matrix_result_manual[i]) > CALCULUS_ERROR_MAX) {
      if (error_count < 10)
        printf("Got %f while expected %f\n", matrix_result_manual[i], matrix_result_auto[i]);
      error_count++;
    }
  }

  printf("%llu ticks for AUTO\n", time_auto_end - time_auto_start);
  printf("%llu ticks for MANUAL\n", time_manual_end - time_manual_start);
  printf("%d errors for MANUAL\n", error_count);
}


