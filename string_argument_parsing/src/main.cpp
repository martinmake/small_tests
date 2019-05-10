#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	(void) argc;

	std::string args = argv[1];

	for (size_t first = args.find_first_of('"'), second; first != std::string::npos; first = args.find_first_of('"')) {
		second = args.find_first_of('"', first + 1);

		if (second == std::string::npos) {
			std::cout << "[-] EXPECTED SYMBOL '\"`" << std::endl;
			exit(1);
		}

		std::string arg = args.substr(first + 1, second - first - 1);
		args.erase(first, second - first + 1);

		std::cout << arg << std::endl;
	}

	std::stringstream args_stream(args);
	for (std::string arg; args_stream >> arg; )
		std::cout << arg << std::endl;

	return 0;
}
