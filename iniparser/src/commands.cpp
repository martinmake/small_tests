#include <iostream>
#include <sstream>

#include "commands.h"

std::string examined_section = "";

void list(const std::vector<std::string>& args)
{
	if (args.empty()) {
		int section_count = ini.get_section_count();
		for (int i = 0; i < section_count; i++)
			printf("[%s]\n", ini.get_section_name(i).c_str());
	} else {
		for (std::string arg; arg = args_stream.pop_front(); ) {
			std::vector<std::string> keys = ini.get_section_keys(arg);
			printf("[%s]\n", arg.c_str());
			for (std::string key : keys)
				std::cout << key << " = " << ini.get_string(arg, key) << std::endl;
		}
	}
}

void examine(const std::vector<std::string>& args)
{
	if (args.empty() || args.size() > 1)
		return;
}

void exit(const std::vector<std::string>& args)
{
	(void) args;

	exit(0);
}

std::map<const std::string, command_func> commands = {
	{ "ls",      list    },
	{ "examine", examine },
	{ "exit",    exit    }
};
