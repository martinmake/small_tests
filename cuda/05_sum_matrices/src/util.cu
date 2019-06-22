#include <stdio.h>

#include <cuda_runtime.h>

#include "util.h"

void cuda_select_device(int device)
{
	cudaDeviceProp deviceProp;
	cudaCall(cudaGetDeviceProperties(&deviceProp, device));
	printf("[DEVICE:%d]: %s\n", device, deviceProp.name);
	cudaCall(cudaSetDevice(device));
}
