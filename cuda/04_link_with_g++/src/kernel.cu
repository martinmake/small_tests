#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define ELEMENTS (0xFF                    )
#define BYTES    (ELEMENTS * sizeof(float))

void initial_data(float* p, size_t size)
{
	for (float* ip = p + size - 1; ip >= p; ip--)
		*ip = (rand() % 0xFF) / 10.0;
}

__global__ void sum_arrays(float* A, float* B, float* C)
{
	C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

void do_cuda_stuff(void)
{
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	h_A = (float*) malloc(BYTES);
	h_B = (float*) malloc(BYTES);
	h_C = (float*) malloc(BYTES);
	cudaMalloc(&d_A, BYTES);
	cudaMalloc(&d_B, BYTES);
	cudaMalloc(&d_C, BYTES);

	srand(time(NULL));

	initial_data(h_A, ELEMENTS);
	initial_data(h_B, ELEMENTS);

	cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);
	sum_arrays<<<1, ELEMENTS>>>(d_A, d_B, d_C);
	cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < ELEMENTS; i++)
		printf("%f = %f + %f;\n", h_C[i], h_A[i], h_B[i]);

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaDeviceReset();
}
