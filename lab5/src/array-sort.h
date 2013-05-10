/*
 * array-sort.h
 *
 *  Created on: May 10, 2013
 *      Author: keritaf
 */

#ifndef ARRAY_SORT_H_
#define ARRAY_SORT_H_

typedef unsigned int uint;

__global__ void bitonicSortShard2(float* devKey, uint countDir, uint count, uint dir);
__global__ void bitonicSortShard(float* devKey, uint count, uint dir);
__global__ void bitonicSortShardDir(float* devKey, uint dir);


#endif /* ARRAY_SORT_H_ */
