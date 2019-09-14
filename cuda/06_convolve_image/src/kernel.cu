#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <array>

#include <opencv2/core.hpp>

#include <cuda_runtime.h>

#include "kernel.h"
#include "util.h"

#define BLOCK_DIMX (0x20)
#define BLOCK_DIMY (0x20)

__global__ void convolve(uint8_t* image, uint8_t* convolved_image, float* global_kernel, uint8_t knx, uint8_t kny, uint32_t nx)
{
	uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t i = (iy * (nx - 2) + ix) * 3;

	__shared__ float kernel[kny * knx];

	if (threadIdx.x < 3 && threadIdx.y < 3)
		kernel[threadIdx.y * knx + threadIdx.x] = global_kernel[threadIdx.y * knx + threadIdx.x];

	__syncthreads();

	for (uint8_t color_offset = 0; color_offset < 3; color_offset++)
	{
		const uint32_t convolved_image_index = i + color_offset;
		uint16_t color = 0;
		for (int8_t y = 0; y < kny; y++)
		{
			for (int8_t x = 0; x < knx; x++)
			{
				const uint32_t image_index = i + color_offset + (y * nx + x) * 3;
				color += image[image_index] * kernel[y][x];
			}
		}
		convolved_image[convolved_image_index] = color / (kny * knx);
	}
}

void cuda_convolve(cv::Mat& image)
{
	uint32_t nx = image.size().width;
	uint32_t ny = image.size().height;

	uint32_t size           = (nx - 0) * (ny - 0) * 3;
	uint32_t convolved_size = (nx - 2) * (ny - 2) * 3;

	cv::Mat convolved_image(ny, nx, CV_8UC3, cv::Scalar(0, 0, 0));

	uint8_t* h_image;
	uint8_t* d_image;
	uint8_t* h_convolved_image;
	uint8_t* d_convolved_image;

	h_image           = image.data;
	h_convolved_image = convolved_image.data;

	dim3 block(BLOCK_DIMX, BLOCK_DIMY);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	cudaCall(cudaMalloc(&d_image,           size));
	cudaCall(cudaMalloc(&d_convolved_image, size));

	cudaCall(cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice));
	convolve<<<grid, block>>>(d_image, d_convolved_image, nx);
	cudaCall(cudaMemcpy(h_convolved_image, d_convolved_image, convolved_size, cudaMemcpyDeviceToHost));

	image = convolved_image.clone();

	cudaCall(cudaFree(d_image));
	cudaCall(cudaFree(d_convolved_image));

	cudaCall(cudaDeviceReset());
}
