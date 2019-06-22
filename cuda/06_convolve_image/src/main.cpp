#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

#include "kernel.h"
#include "util.h"

#define WINDOW_NAME "IMAGE WINDOW"

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		std::cout << "Usage: " << argv[0] << " <path_to_image> <convolutions>" << std::endl;
		exit(1);
	}

	int convolutions = std::stoi(argv[2]);

	cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

	if (image.empty())
	{
		std::cout << "COULD NOT OPEN OR FIND THE IMAGE" << std::endl;
		exit(1);
	}

	cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

	cv::imshow(WINDOW_NAME, image);
	cv::waitKey(0);

	cuda_select_device(0);
	for (uint8_t i = 0; i < convolutions; i++)
		cuda_convolve(image);

	cv::imshow(WINDOW_NAME, image);
	cv::waitKey(0);

	return 0;
}
