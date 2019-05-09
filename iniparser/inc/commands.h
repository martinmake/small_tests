#ifndef _INIPARSER_COMMANDS_H_
#define _INIPARSER_COMMANDS_H_

#include <map>
#include <string>

#include "ini.h"

using command_func = void (*)(const std::string& argv);

extern std::map<const std::string, command_func> commands;
extern Ini ini;

extern void list(void);
extern void examine(std::string& argv);
extern void exit();

#endif
