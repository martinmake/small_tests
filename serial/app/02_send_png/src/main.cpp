#include <iostream>
#include <fstream>

#define FTDI_PATH "/dev/ttyUSB0"
#define IMG_PATH  "res/arch.png"

int main(void)
{
	std::ofstream ftdi (FTDI_PATH, std::ofstream::binary);
	std::ifstream image(IMG_PATH,  std::ifstream::binary);

	ftdi << image.rdbuf();

	return 0;
}
