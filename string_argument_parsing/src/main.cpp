#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	if (argc != 2)
		exit(1);

	std::string args = argv[1];

	if (std::count(args.begin(), args.end(), '"') % 2 != 0) {
		std::cout << "[-] EXPECTED ‘\"’" << std::endl;
		exit(1);
	}

	args.erase(0, args.find_first_not_of(' '));
	args.erase(args.find_last_not_of(' ') + 1);
	char stopper = args[0] == '"' ? '"' : ' ';

 	for (std::string::iterator it = args.begin() + (stopper == '"' ? 1 : 0); it < args.end(); it++) {
		static std::string arg = "";

		if (*it == stopper) {
			it++;
			while (*it == ' ')
				it++;

			stopper = *it == '"' ? '"' : ' ';

			std::cout << arg << std::endl;

			arg.clear();
		} else {
			arg.push_back(*it);
		}
 	}

	return 0;
}
