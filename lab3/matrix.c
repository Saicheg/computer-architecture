#include "matrix.h"

long matrix_element_size;

float * matrix_a;
float * matrix_b;
float * matrix_result_1;
float * matrix_result_2;

#define A(i,j) matrix_a[i*MATRIX_SIDE+j]
#define B(i,j) matrix_b[i*MATRIX_SIDE+j]
#define R1(i,j) matrix_result_1[i*MATRIX_SIDE+j]
#define R2(i,j) matrix_result_2[i*MATRIX_SIDE+j]

long cache_l1_size, cache_l1_assoc, cache_l1_line,
cache_l2_size, cache_l2_assoc, cache_l2_line,
cache_l3_size, cache_l3_assoc, cache_l3_line;
int cache_l3_exists = 0;

long cache_l3_block_size, cache_l2_block_size, cache_l1_block_size;

float* createMatrix(int randomize);
void computeMatrix();
void calculateBlockL3(int i_start, int i_end,
                      int j_start, int j_end,
                      int k_start, int k_end);
void calculateBlockL2(int i_start, int i_end,
                      int j_start, int j_end,
                      int k_start, int k_end);
void calculateBlockL1(int i_start, int i_end,
                      int j_start, int j_end,
                      int k_start, int k_end);

long min(long a, long b) {
    return(a < b) ? a : b;
}

unsigned long long int rdtsc() {
    unsigned long long int c, d;
    asm volatile("rdtsc" : "=a" (c), "=d" (d)); //assembly code running the instruction
    return c | (d << 32); // calculating the tick value.
}

int main() {
    matrix_element_size = sizeof(float);

    printf("Size of float: %ld\n\n", sizeof(float));

    printf("Querying cache parameters... ");
    cache_l1_size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    cache_l1_assoc = sysconf(_SC_LEVEL1_DCACHE_ASSOC);
    cache_l1_line = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    cache_l2_size = sysconf(_SC_LEVEL2_CACHE_SIZE);
    cache_l2_assoc = sysconf(_SC_LEVEL2_CACHE_ASSOC);
    cache_l2_line = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
    cache_l3_size = sysconf(_SC_LEVEL3_CACHE_SIZE);
    cache_l3_assoc = sysconf(_SC_LEVEL3_CACHE_ASSOC);
    cache_l3_line = sysconf(_SC_LEVEL3_CACHE_LINESIZE);

    if (cache_l3_size != 0) {
        cache_l3_exists = 1;
    }
    printf("done.\n\n");

    printf("Cache settings:\n");
    printf("  L1: %ld KiB, %ld lines, line size of %ld bytes, %ld-associative\n",
           cache_l1_size / 1024,
           cache_l1_size / cache_l1_line,
           cache_l1_line,
           cache_l1_assoc);
    printf("  L2: %ld KiB, %ld lines, line size of %ld bytes, %ld-associative\n",
           cache_l2_size / 1024,
           cache_l2_size / cache_l2_line,
           cache_l2_line,
           cache_l2_assoc);
    if (cache_l3_exists == 0) {
        printf("  L3: no cache\n");
    } else {
        printf("  L3: %ld KiB, %ld lines, line size of %ld bytes, %ld-associative\n",
               cache_l3_size / 1024,
               cache_l3_size / cache_l3_line,
               cache_l3_line,
               cache_l3_assoc);
    }

    cache_l1_block_size = cache_l1_size / matrix_element_size / cache_l1_line / 3;
    cache_l2_block_size = cache_l2_size / matrix_element_size / cache_l2_line / 3;
    if (cache_l3_exists != 0) {
        cache_l3_block_size = cache_l3_size / matrix_element_size / cache_l3_line / 3;
    }

    printf("\nBlock sizes:\n");
    printf("  L1: %ld x %ld\n", cache_l1_line, cache_l1_block_size);
    printf("  L2: %ld x %ld\n", cache_l2_line, cache_l2_block_size);
    if (cache_l3_exists != 0) {
        printf("  L3: %ld x %ld\n", cache_l3_line, cache_l3_block_size);
    }

    srand(time(NULL));

    printf("\nAllocating argument matrixes... ");
    matrix_a = createMatrix(1);
    matrix_b = createMatrix(1);
    printf("done.\n");

    unsigned long long time_auto_start, time_auto_end,
        time_manual_start, time_manual_end;

    printf("Allocating result 1 matrix... ");
    matrix_result_1 = createMatrix(0);
    printf("done. \nComputing result 1... ");
    fflush(stdout);
    time_auto_start = rdtsc();
    computeMatrix(matrix_a, matrix_b, matrix_result_1);
    time_auto_end = rdtsc();

    printf("done. \nAllocating result 2 matrix...");
    matrix_result_2 = createMatrix(0);
    printf("done. \nComputing result 2... ");
    fflush(stdout);
    time_manual_start = rdtsc();
    if (cache_l3_exists != 0) {
        calculateBlockL3(0, MATRIX_SIDE,
                         0, MATRIX_SIDE,
                         0, MATRIX_SIDE);
    } else {
        calculateBlockL2(0, MATRIX_SIDE,
                         0, MATRIX_SIDE,
                         0, MATRIX_SIDE);
    }
    time_manual_end = rdtsc();
    printf("done.\n\n");


    free(matrix_a);
    free(matrix_b);

    long error_count = 0;
    for (long i = 0; i < MATRIX_TOTAL_ELEMENTS; i++) {
        if (fabs(1 - matrix_result_1[i] / matrix_result_2[i]) > (MATRIX_MAX_ERROR_PERCENT / 100.0)) {
            if (error_count < 10)
                printf("Got %f while expected %f\n", matrix_result_2[i], matrix_result_1[i]);
            error_count++;
        }
    }

    printf("Result 1: %llu ticks\n", time_auto_end - time_auto_start);
    printf("Result 2: %llu ticks\n", time_manual_end - time_manual_start);
    printf("%ld of total %ld differs in values\n",
           error_count,
           (long) (MATRIX_TOTAL_ELEMENTS));

    free(matrix_result_1);
    free(matrix_result_2);

    return 0;
}

float* createMatrix(int randomize) {
    float *matrix = (float*) memalign(64, sizeof(float) * MATRIX_TOTAL_ELEMENTS);
    if (randomize != 0) {
        for (int i = 0; i < MATRIX_TOTAL_ELEMENTS; i++) {
            matrix[i] = (float) rand() / (float) MATRIX_MAX_FLOAT;
        }
    } else {
        for (int i = 0; i < MATRIX_TOTAL_ELEMENTS; i++) {
            matrix[i] = 0.0;
        }

    }
    return matrix;
}

void freeMatrix(float ** matrix) {
    free(*matrix);
    (*matrix) = NULL;
}

void computeMatrix() {
    for (int i = 0; i < MATRIX_SIDE; i++) {
        for (int j = 0; j < MATRIX_SIDE; j++) {
            for (int k = 0; k < MATRIX_SIDE; k++) {
                R1(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

void calculateBlockL3(int i_start, int i_end,
                      int j_start, int j_end,
                      int k_start, int k_end) {
    for (int i = i_start; i < min(i_end, i + cache_l3_line); i += cache_l3_line) {
        for (int j = j_start; j < min(j_end, j + cache_l3_line); j += cache_l3_line) {
            for (int k = k_start; k < min(k_end, k + cache_l3_block_size); k += cache_l3_block_size) {
                calculateBlockL2(i, min(i_end, i + cache_l3_line),
                                 j, min(j_end, j + cache_l3_line),
                                 k, min(k_end, k + cache_l3_block_size));
            }
        }
    }
}

void calculateBlockL2(int i_start, int i_end,
                      int j_start, int j_end,
                      int k_start, int k_end) {
    for (int i = i_start; i < min(i_end, i + cache_l2_line); i += cache_l2_line) {
        for (int j = j_start; j < min(j_end, j + cache_l2_line); j += cache_l2_line) {
            for (int k = k_start; k < min(k_end, k + cache_l2_block_size); k += cache_l2_block_size) {
                calculateBlockL1(i, min(i_end, i + cache_l2_line),
                                 j, min(j_end, j + cache_l2_line),
                                 k, min(k_end, k + cache_l2_block_size));
            }
        }
    }
}

void calculateBlockL1(int i_start, int i_end,
                      int j_start, int j_end,
                      int k_start, int k_end) {
    for (int i = i_start; i < i_end; i += cache_l1_line) {
        for (int j = j_start; j < j_end; j += cache_l1_line) {
            for (int k = k_start; k < k_end; k += cache_l1_block_size) {
                int ii_end = min(i + cache_l1_line, i_end);
                int jj_end = min(j + cache_l1_line, j_end);
                int kk_end = min(k + cache_l1_block_size, k_end);
                for (int ii = i; ii < ii_end; ii++) {
                    for (int jj = j; jj < jj_end; jj++) {
                        for (int kk = k; kk < kk_end; kk++) {
                            R2(ii,jj) += A(ii, kk) * B(kk,jj);
                        }
                    }
                }
            }

        }
    }

}
