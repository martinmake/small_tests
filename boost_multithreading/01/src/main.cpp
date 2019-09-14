#include <iostream>
#include <string>
#include <vector>
#include <boost/thread.hpp>

void print(const std::vector<std::string>& strings)
{
	for (const std::string& string : strings)
		std::cout << string << std::endl;
}

int main(void)
{
	std::vector<std::string> upper_case_strings = { "AAAAAAAAAAAAA!", "BBBBBBBBBBBBB!", "CCCCCCCCCCCCC!", "DDDDDDDDDDDDD!", "EEEEEEEEEEEEE!", "FFFFFFFFFFFFF!" };
	std::vector<std::string> lower_case_strings = { "aaaaaaaaaaaaa!", "bbbbbbbbbbbbb!", "ccccccccccccc!", "ddddddddddddd!", "eeeeeeeeeeeee!", "fffffffffffff!" };

	boost::thread t(print, upper_case_strings);

	print(lower_case_strings);
	print(lower_case_strings);
	print(lower_case_strings);
	print(lower_case_strings);
	print(lower_case_strings);
	print(lower_case_strings);
	print(lower_case_strings);
	print(lower_case_strings);

	t.join();

	return 0;
}
