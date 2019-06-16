#include <stdio.h>
#include <cuda_runtime.h>

__global__ void check_index(void)
{
	printf("threadIdx: %d, %d, %d\n"
	       "blockIdx:  %d, %d, %d\n"
	       "blockDim:  %d, %d, %d\n"
	       "gridDim:   %d, %d, %d\n\n",
	       threadIdx.x, threadIdx.y, threadIdx.z,
	       blockIdx.x,  blockIdx.y,  blockIdx.z,
	       blockDim.x,  blockDim.y,  blockDim.z,
	       gridDim.x,   gridDim.y,   gridDim.z);
}

int main(void)
{
	check_index<<<1, 7>>>();

	cudaDeviceReset();
	return 0;
}
