#ifndef _LAB4_MATRIX_H
#define _LAB4_MATRIX_H

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

// Matrix type definition
struct Matrix {
	int width;
	int height;
	float* elements;
	int stride;
};

// Thread block size
#define CUDA_BLOCK_SIZE 16

#define MATRIX_MAX_FLOAT 100
#define MATRIX_MAX_ERROR_PERCENT 1
#endif
