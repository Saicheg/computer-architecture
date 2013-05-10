#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <time.h>
using namespace std;

#include "array-sort.h"

#define PROFILING_CLOCK CLOCK_PROCESS_CPUTIME_ID

timespec time_current() {
	timespec temp;
	clock_gettime(PROFILING_CLOCK, &temp);
	return temp;
}

timespec time_diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

uint nextPowerOfTwo(uint x)
{
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;

    //return 1U << (W - __clz(x - 1));
}

int main()
{
	uint n, n_ext;
	float *f_hostKey, *f_devKey, *f_hostKeyCpu;

	uint direction = 1;

	curandStatus_t cuRStatus;
	curandGenerator_t cuRGen;

	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float gpuTime;

	printf("Input N (max 33554432): ");
	scanf("%u", &n);
	if(n <= 0) return 1;
	n_ext = nextPowerOfTwo(n);

	printf("Sort ascending? Y/N\n");
	char c = '\0';
	do {
		fflush(stdin);
		c = getchar();
	} while (c !='y' && c != 'Y' && c != 'n' && c != 'N');
	if(c == 'y' || c == 'Y') direction = 1;
	else direction = 0;

	
	f_hostKey = (float*)malloc(sizeof(float) * n);
	f_hostKeyCpu = (float*)malloc(sizeof(float) * n);
	cudaMalloc((void**)&f_devKey, n_ext * sizeof(float));

	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);

	cuRStatus = curandCreateGenerator(&cuRGen,CURAND_RNG_PSEUDO_DEFAULT);
	if(cuRStatus != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "curandCreateGenerator failed!\n");
		return 1;
	}

	cuRStatus = curandSetPseudoRandomGeneratorSeed(cuRGen, 1234ULL);
	if(cuRStatus != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "curandSetPseudoRandomGeneratorSeed failed!\n");
		return 1;
	}

	cuRStatus = curandGenerateUniform(cuRGen, f_devKey, n);
	if(cuRStatus != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "curandGenerateUniform failed!\n");
		return 1;
	}
	
	cudaMemcpy(f_hostKeyCpu, f_devKey, n * 4, cudaMemcpyDeviceToHost);
	timespec startCpuTime, endCpuTime, diffCpuTime;
	float cpuTime;
	float temp;

	printf("\nCPU:\n");
	startCpuTime = time_current();
	if(direction)
	{
		for(int i = 0; i < n; i++)
		{
			for(int j = n - 1; j > i; j--)
			{			
				if(f_hostKeyCpu[j] < f_hostKeyCpu[i])
				{
					temp = f_hostKeyCpu[j];
					f_hostKeyCpu[j] = f_hostKeyCpu[i];
					f_hostKeyCpu[i] = temp;
				}
			}
		}
	}
	else
	{
		for(int i = 0; i < n; i++)
		{
			for(int j = n - 1; j > i; j--)
			{			
				if(f_hostKeyCpu[j] > f_hostKeyCpu[i])
				{
					temp = f_hostKeyCpu[j];
					f_hostKeyCpu[j] = f_hostKeyCpu[i];
					f_hostKeyCpu[i] = temp;
				}
			}
		}
	}

	endCpuTime = time_current();
	diffCpuTime = time_diff(startCpuTime, endCpuTime);
	cpuTime = diffCpuTime.tv_sec * 1000.0 + diffCpuTime.tv_sec / 1e6;
	printf("Time sorting: %f ms\n\n", cpuTime);

	printf("\nGPU:\n");
	int blockDim = 512;
	int gridDim = ((n_ext / blockDim) / 2);
	if(gridDim == 0) gridDim = 1;
	if(n_ext <= 512) blockDim = n_ext / 2;
	
	cudaEventRecord(start, 0);
	bitonicSortShardDir<<<gridDim, blockDim, blockDim * 2 * sizeof(uint)>>>(f_devKey, direction);
	cudaDeviceSynchronize();
	for(int i = 2; i <= gridDim; i *= 2)
	{
		for(int j = i; j > 1; j /= 2)
		{
			bitonicSortShard2<<<gridDim, blockDim>>>(f_devKey, i, j, direction);
			cudaDeviceSynchronize();
		}

		bitonicSortShard<<<gridDim, blockDim, blockDim * 2 * sizeof(uint)>>>(f_devKey, i, direction);
		cudaDeviceSynchronize();
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	printf("Time sorting: %f ms\n", gpuTime);

	if(direction)
		cudaMemcpy(f_hostKey, f_devKey + n_ext - n, n * sizeof(uint), cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(f_hostKey, f_devKey, n * sizeof(uint), cudaMemcpyDeviceToHost);

	int correctFlag = 1;
	printf("\nCheck: ");
	for(int j = 0; j < n; j++)
		if(f_hostKey[j] != f_hostKeyCpu[j])
		{
			correctFlag = 0;
			break;
		}
	/*if(direction)
	{
		for(int j = 1; j < n; j++)
			if(f_hostKey[j - 1] > f_hostKey[j])
			{
				correctFlag = 0;
				break;
			}
	}
	else
	{
		for(int j = 1; j < n; j++)
			if(f_hostKey[j - 1] < f_hostKey[j])
			{
				correctFlag = 0;
				break;
			}
	}*/
	printf(correctFlag ? "OK\n" : "Failed!\n");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	curandDestroyGenerator(cuRGen);
	cudaFree(f_devKey);
	free(f_hostKey);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
	{
        return 1;
    }

	system("pause");
    return 0;
}
