#include <iostream>
#include <string>
#include <sstream>
#include <signal.h>
#include <stdlib.h>

#include "ini.h"
#include "commands.h"

#define INI_PATH_INPUT  "inis/input.ini"
#define INI_PATH_OUTPUT "inis/output.ini"

Ini ini(INI_PATH_INPUT);

static void finish(int sig)
{
	(void) sig;

	exit(0);
}

int main(void)
{
	signal(SIGINT, finish);

	while (1) {
		std::string       line;
		std::stringstream line_stream;
		std::string       statement;
		std::string       command; std::string       argv;

		std::cout << "> ";

		getline(std::cin, line);
		line_stream << line << ';';

		while (getline(line_stream, statement, ';')) {
			statement.erase(0, statement.find_first_not_of(' '));

			if (statement.find_first_of(' ') == std::string::npos) {
				command = statement;
				argv    = "";
			} else {
				command = statement.substr(0, statement.find_first_of(' '));
				argv    = statement.substr(command.size(), statement.size());
				argv.erase(0, argv.find_first_not_of(' '));
				if (argv.find_last_of(' ') != std::string::npos)
					argv.erase(argv.find_last_of(' '), argv.size());
			}

			if (commands.find(command)->second)
				commands.find(command)->second(argv);
			else
				std::cout << "[-] COMMAND NOT FOUND" << std::endl;
		}
	}

	ini.dump(INI_PATH_OUTPUT);

	return 0;
}
