#include <iostream>
#include <boost/filesystem.hpp>

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
		exit(1);
	}

	boost::filesystem::path path(argv[1]);

	if (boost::filesystem::exists(path))
		std::cout << "FILE " << path << " EXISTS." << std::endl;
	else
		std::cout << "FILE " << path << " DOES NOT EXIST." << std::endl;


	return 0;
}
