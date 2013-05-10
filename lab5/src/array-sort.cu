#include "array-sort.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <time.h>

/*


bitonicSortShardDir - разбиение всей последовательности на битонические.
						четные блоки сортируют в одном направлении, нечетные в другом.
bitonicSortShard2   - объединение двух битонических последовательностей в одну
		четные блоки сортируют в одном направлении, нечетные в другом.
		один блок обрабатывает block_size * 2 элементов
		count - кол-во таких блоков для объедининия в один большой блок
		countDir - кол-во блоков для объедининия в один большой блок. задает одинаковое направление сортировки
		в одном большом блоке при работе с меньшими блоками
bitonicSortShard	- сортирует блок
		count - кол-во блоков которые были объединены на предыдущем шаге
				и сейчас требуют сортировки элементов в них в одном направлении
*/

__device__ void Comparator(float &keyA, float &keyB, uint dir)
{
	float t;
	if( (keyA > keyB) == dir)
	{
		t = keyA; keyA = keyB; keyB = t;
	}
}

__global__ void bitonicSortShardDir(float* devKey, uint dir)
{
	extern __shared__ float sk[];
	int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	
	sk[threadIdx.x] = devKey[index];
	sk[threadIdx.x + blockDim.x] = devKey[index + blockDim.x];

	for(uint size = 2; size <= blockDim.x * 2; size <<= 1)
	{
		uint ddd =(dir ^ (( threadIdx.x & (size / 2)) != 0) ) ^ (blockIdx.x & 1);
		
		for(uint stride = size >> 1; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint pos = 2 * threadIdx.x - ( threadIdx.x & (stride - 1));
			Comparator(sk[pos], sk[pos + stride], ddd);
		}
	}

	__syncthreads();

	devKey[index] = sk[threadIdx.x];
	devKey[index + blockDim.x] = sk[threadIdx.x + blockDim.x];
}

__global__ void bitonicSortShard2(float* devKey, uint countDir, uint count, uint dir)
{
	int num = (blockIdx.x - blockIdx.x % count) / count;
	uint ddd = (((blockIdx.x - blockIdx.x % countDir) / countDir) % 2) ^ dir;
	uint stride = blockDim.x * count;
	uint pos = threadIdx.x + blockIdx.x * blockDim.x + blockDim.x * count * num;
	Comparator(devKey[pos], devKey[pos + stride], ddd);
}

__global__ void bitonicSortShard(float* devKey, uint count, uint dir)
{
	extern __shared__ float sk [];
	int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	sk[threadIdx.x] = devKey[index];
	sk[threadIdx.x + blockDim.x] = devKey[index + blockDim.x];

	int num = (blockIdx.x - blockIdx.x % count) / count;
	uint ddd = (num % 2) ^ dir;
	for(uint stride = blockDim.x; stride > 0; stride >>= 1)
	{
		__syncthreads();
		uint pos = 2 * threadIdx.x - ( threadIdx.x & (stride - 1));
		Comparator(sk[pos], sk[pos + stride], ddd);
	}

	__syncthreads();

	devKey[index] = sk[threadIdx.x];
	devKey[index + blockDim.x] = sk[threadIdx.x + blockDim.x];
}
