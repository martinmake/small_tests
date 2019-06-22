#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

#define WINDOW_NAME "IMAGE WINDOW"

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
		exit(1);
	}

	cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

	if (image.empty())
	{
		std::cout << "COULD NOT OPEN OR FIND THE IMAGE" << std::endl;
		exit(1);
	}

	cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

	cv::imshow(WINDOW_NAME, image);
	cv::waitKey(0);

	for (uint16_t y = 0; y < image.size().height * 3; y++)
	{
		for (uint16_t x = 0; x < image.size().width * 3; x++)
		{
			uint32_t idx = image.size().width * y + x;
			if (idx % 3 == 0)
				*(image.data + idx) = 255;
		}
	}

	cv::imshow(WINDOW_NAME, image);
	cv::waitKey(0);

	return 0;
}
