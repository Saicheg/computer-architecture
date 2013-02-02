#include "matrix.h"
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

float *** matrix_a;
float *** matrix_b;
float *** matrix_result;
int i, j, k;

int main() {
  prepareMatrixes();

  freeMatrix(&matrix_a);
  freeMatrix(&matrix_b);
  freeMatrix(&matrix_result);
  return 0;
}

void prepareMatrixes() {
  srand(time(NULL));

  createMatrix(&matrix_a);
  createMatrix(&matrix_b);
  createMatrix(&matrix_result);
}

void createMatrix(float **** matrix) {
  (*matrix) = (float***) calloc(N, sizeof(float**));
  for (i = 0; i < N; i++) {
    (*matrix)[i] = (float**) calloc(N, sizeof(float*));
    for (j = 0; j < N; j++) {
      (*matrix)[i][j] =  (float*) calloc(K*K, sizeof(float));
      for (k = 0; k < K*K; k++) {
        (*matrix)[i][j][k] = (float) rand() / (float) FLOATMAX;
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
