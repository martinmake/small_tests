#include <stdio.h>

#include "kernel.h"
#include "util.h"

int main(void)
{
	cuda_select_device(0);

	getchar();

	cuda_work();

	return 0;
}
