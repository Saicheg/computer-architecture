#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

unsigned long long int rdtsc() {
	unsigned long long int c, d;
	asm volatile("rdtsc" : "=a" (c), "=d" (d));
	//assembly code running the instruction
	return c | (d << 32); // calculating the tick value.
}

// Get a matrix element
__device__ float GetMatrixElement(const Matrix A, int row, int col) {
	return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetMatrixElement(Matrix A, int row, int col, float value) {
	A.elements[row * A.stride + col] = value;
}
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
	Matrix Asub;
	Asub.width = CUDA_BLOCK_SIZE;
	Asub.height = CUDA_BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * CUDA_BLOCK_SIZE * row
			+ CUDA_BLOCK_SIZE * col];
	return Asub;
}
__global__ void CudaMatrixMultiplicationKernel(Matrix A, Matrix B, Matrix C) {
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	Matrix C_submatrix = GetSubMatrix(C, blockRow, blockCol);
	float sum = 0.0;
	int row = threadIdx.y;
	int col = threadIdx.x;
	for (int m = 0; m < (A.width / CUDA_BLOCK_SIZE); ++m) {
		Matrix A_submatrix = GetSubMatrix(A, blockRow, m);
		Matrix B_submatrix = GetSubMatrix(B, m, blockCol);
		__shared__ float As[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
		__shared__ float Bs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
		As[row][col] = GetMatrixElement(A_submatrix, row, col);
		Bs[row][col] = GetMatrixElement(B_submatrix, row, col);
		__syncthreads();
		for (int e = 0; e < CUDA_BLOCK_SIZE; ++e)
			sum += As[row][e] * Bs[e][col];
		__syncthreads();
	}
	SetMatrixElement(C_submatrix, row, col, sum);
}

void CudaMatrixMultiplication(const struct Matrix A, const struct Matrix B,
		struct Matrix C) {
	struct Matrix d_A;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	CUDA_CHECK_RETURN(cudaMalloc(&d_A.elements, size))
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice))

	Matrix d_B;
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	CUDA_CHECK_RETURN(cudaMalloc(&d_B.elements, size))
	CUDA_CHECK_RETURN(
			cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));

	Matrix d_C;
	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	CUDA_CHECK_RETURN(cudaMalloc(&d_C.elements, size));

	dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	CudaMatrixMultiplicationKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	CUDA_CHECK_RETURN(cudaThreadSynchronize())

	CUDA_CHECK_RETURN(
			cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost))

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

void CpuMatrixMultiplication(const struct Matrix A, const struct Matrix B,
		struct Matrix C) {
	for (int i = 0; i < C.height; i++) {
		for (int j = 0; j < C.width; j++) {
			float sum = 0.0;
			for (int k = 0; k < A.width; k++) {
				sum += A.elements[i * A.stride + k]
						* B.elements[k * B.stride + j];
			}
			C.elements[i * C.stride + j] = sum;
		}
	}
}

int main(int argc, char* argv[]) {
	srand(time(NULL));
	if (argc < 4) {
		printf("Call with arguments: <a height> <a width> <b width>\n");
		exit(0);
	}

	int a1, a2, b1, b2;
	a1 = atoi(argv[1]); /* Height of A */
	a2 = atoi(argv[2]); /* Width of A */
	b1 = a2; /* Height of B */
	b2 = atoi(argv[3]); /* Width of B */

	Matrix A, B, C_cuda, C_cpu;

	A.height = a1;
	A.width = A.stride = a2;
	A.elements = (float*) malloc(A.width * A.height * sizeof(float));

	B.height = b1;
	B.width = B.stride = b2;
	B.elements = (float*) malloc(B.width * B.height * sizeof(float));

	C_cuda.height = A.height;
	C_cuda.width = C_cuda.stride = B.width;
	C_cuda.elements = (float*) malloc(
			C_cuda.width * C_cuda.height * sizeof(float));

	C_cpu.height = A.height;
	C_cpu.width = C_cpu.stride = B.width;
	C_cpu.elements = (float*) malloc(
			C_cpu.width * C_cpu.height * sizeof(float));

	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			A.elements[i * A.width + j] = (float) (rand() % MATRIX_MAX_FLOAT);
	for (int i = 0; i < B.height; i++)
		for (int j = 0; j < B.width; j++)
			B.elements[i * B.width + j] = (float) (rand() % MATRIX_MAX_FLOAT);

	unsigned long long time_start_cpu, time_stop_cpu, time_start_cuda,
			time_stop_cuda;

	time_start_cpu = rdtsc();
	CpuMatrixMultiplication(A, B, C_cpu);
	time_stop_cpu = rdtsc();
	time_start_cuda = rdtsc();
	CudaMatrixMultiplication(A, B, C_cuda);
	time_stop_cuda = rdtsc();

	long error_count = 0;
	for (long i = 0; i < C_cpu.width * C_cpu.height; i++) {
		if (fabs((float) (1 - C_cuda.elements[i] / C_cpu.elements[i]))
				> (MATRIX_MAX_ERROR_PERCENT / 100.0)) {
			if (error_count < 10)
				printf("Got %f while expected %f\n", C_cuda.elements[i],
						C_cpu.elements[i]);
			error_count++;
		}
	}

	unsigned long long time_cpu = time_stop_cpu - time_start_cpu, time_cuda =
			time_stop_cuda - time_start_cuda;
	float ratio = (float) time_cpu / (float) time_cuda;
	printf("CPU:  %15llu ticks\n", time_cpu);
	printf("CUDA: %15llu ticks\n", time_cuda);
	printf("CPU/CUDA ratio is %0.2f\n", ratio);
	printf("%ld of total %ld values differ\n", error_count,
			(long) (C_cpu.width * C_cpu.height));

	free(A.elements);
	free(B.elements);
	free(C_cpu.elements);
	free(C_cuda.elements);
}
