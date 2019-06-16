#include <stdio.h>

extern void hello_from_cpu(void);

__global__ void hello_from_gpu(void)
{
	printf("HELLO FROM GPU\n");
}

int main(void)
{
	hello_from_cpu();
	hello_from_gpu<<<1, 10>>>();

	cudaDeviceReset();
	return 0;
}
