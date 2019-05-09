#include <iostream>

#include "commands.h"

void list(const std::string& argv)
{
	(void) argv;

	int section_cout = ini.get_section_count();

	for (int i = 0; i < section_cout; i++)
		printf("[%s]\n", ini.get_section_name(i));
}

void examine(const std::string& argv)
{
	std::cout << argv << std::endl;
}

void exit(const std::string& argv)
{
	(void) argv;

	exit(0);
}

std::map<const std::string, command_func> commands = {
	{ "ls",      list    },
	{ "examine", examine },
	{ "exit",    exit    }
};
