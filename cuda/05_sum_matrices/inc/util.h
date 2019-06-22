#ifndef _UTIL_H_
#define _UTIL_H_

#define cudaCall(call)                                                                                 \
{                                                                                                      \
	const cudaError_t error = call;                                                                \
	if (error != cudaSuccess)                                                                      \
	{                                                                                              \
		printf("[ERROR:%d]: %s:%d, %s", error, __FILE__, __LINE__, cudaGetErrorString(error)); \
		exit(-10 * error);                                                                     \
	}                                                                                              \
}

extern void cuda_select_device(int device);

#endif
