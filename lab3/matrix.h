#ifndef KERITAF_MATRIX_H
#define KERITAF_MATRIX_H

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

#define MATRIX_SIDE 1000

#define MATRIX_TOTAL_ELEMENTS (MATRIX_SIDE*MATRIX_SIDE)

#define MATRIX_MAX_FLOAT 100
#define MATRIX_MAX_ERROR_PERCENT 0.01

#endif

