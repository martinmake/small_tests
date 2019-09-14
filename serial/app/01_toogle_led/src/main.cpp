#include <iostream>
#include <fstream>
#include <unistd.h>

#define FTDI_PATH "/dev/ttyUSB0"

int main(void)
{
	std::ofstream ftdi(FTDI_PATH);

	while (1) {
		ftdi << '1' << std::flush;
		sleep(1);
		ftdi << '0' << std::flush;
		sleep(1);
	}

	return 0;
}
