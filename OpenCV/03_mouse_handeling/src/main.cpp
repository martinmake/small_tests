#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>

#define SCALE 30

void mouse_event_handler(int event, int x, int y, int flags, void* userdata)
{
	(void) flags;
	(void) userdata;

	x = round(x / SCALE);
	y = round(y / SCALE);

	std::cout << "[EVENT] ";
	switch (event)
	{
		case cv::EVENT_MOUSEMOVE:     std::cout << "MOUSEMOVE";     break;
		case cv::EVENT_LBUTTONDOWN:   std::cout << "LBUTTONDOWN";   break;
		case cv::EVENT_RBUTTONDOWN:   std::cout << "RBUTTONDOWN";   break;
		case cv::EVENT_MBUTTONDOWN:   std::cout << "MBUTTONDOWN";   break;
		case cv::EVENT_LBUTTONUP:     std::cout << "LBUTTONUP";     break;
		case cv::EVENT_RBUTTONUP:     std::cout << "RBUTTONUP";     break;
		case cv::EVENT_MBUTTONUP:     std::cout << "MBUTTONUP";     break;
		case cv::EVENT_LBUTTONDBLCLK: std::cout << "LBUTTONDBLCLK"; break;
		case cv::EVENT_RBUTTONDBLCLK: std::cout << "RBUTTONDBLCLK"; break;
		case cv::EVENT_MBUTTONDBLCLK: std::cout << "MBUTTONDBLCLK"; break;
	}
	std::cout << std::endl;

	std::cout << "[FLAGS] ";
	if (flags & cv::EVENT_FLAG_LBUTTON)  std::cout << "LBUTTON ";
	if (flags & cv::EVENT_FLAG_RBUTTON)  std::cout << "RBUTTON ";
	if (flags & cv::EVENT_FLAG_MBUTTON)  std::cout << "MBUTTON ";
	if (flags & cv::EVENT_FLAG_CTRLKEY)  std::cout << "CTRLKEY ";
	if (flags & cv::EVENT_FLAG_SHIFTKEY) std::cout << "SHIFTKEY ";
	if (flags & cv::EVENT_FLAG_ALTKEY)   std::cout << "ALTKEY ";
	std::cout<< std::endl;

	std::cout << "[POSITION] " << x << ", " << y << std::endl;

	std::cout << std::endl;
}

int main(void)
{
	const std::string winname = "WINDOW";
	cv::Mat image = cv::Mat::zeros(28, 28, CV_8UC1);

	cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(winname, mouse_event_handler, nullptr);

	while (1)
	{
		cv::Mat scaled_image;
		cv::resize(image, scaled_image, cv::Size(), SCALE, SCALE);
		imshow(winname, scaled_image);

		cv::waitKey(0);
	}

	return 0;
}
