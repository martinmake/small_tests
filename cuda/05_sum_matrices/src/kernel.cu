#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <cuda_runtime.h>

#include "util.h"

#define WIDTH  (1 << 10)
#define HEIGHT (1 << 10)

#define ELEMENT_COUNT (WIDTH * HEIGHT               )
#define BYTES         (ELEMENT_COUNT * sizeof(float))

#define BLOCK_DIMX (0x20)
#define BLOCK_DIMY (0x20)

void initial_data(float* p, size_t size)
{
	for (float* ip = p + size - 1; ip >= p; ip--)
		*ip = (rand() % 0xFF) / 10.0;
}

__global__ void sum_matrices(float* A, float* B, float* C)
{
	uint32_t ix = blockIdx.x * BLOCK_DIMX + threadIdx.x;
	uint32_t iy = blockIdx.y * BLOCK_DIMY + threadIdx.y;
	uint32_t i = iy * WIDTH + ix;
	C[i] = A[i] + B[i];
}

void cuda_work(void)
{
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	dim3 block(BLOCK_DIMX, BLOCK_DIMY);
	dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

	h_A = (float*) malloc(BYTES);
	h_B = (float*) malloc(BYTES);
	h_C = (float*) malloc(BYTES);
	cudaCall(cudaMalloc(&d_A, BYTES));
	cudaCall(cudaMalloc(&d_B, BYTES));
	cudaCall(cudaMalloc(&d_C, BYTES));

	srand(time(NULL));

	initial_data(h_A, ELEMENT_COUNT);
	initial_data(h_B, ELEMENT_COUNT);

	cudaCall(cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice));
	cudaCall(cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice));
	sum_matrices<<<grid, block>>>(d_A, d_B, d_C);
	cudaCall(cudaMemcpy(h_C, d_C, BYTES, cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < ELEMENT_COUNT; i++)
		printf("%f = %f + %f;\n", h_C[i], h_A[i], h_B[i]);

	free(h_A);
	free(h_B);
	free(h_C);
	cudaCall(cudaFree(d_A));
	cudaCall(cudaFree(d_B));
	cudaCall(cudaFree(d_C));

	cudaCall(cudaDeviceReset());
}
