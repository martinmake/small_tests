#include "ini.h"

Ini::Ini(const char* path)
{
	m_ini = iniparser_load(path);
}

Ini::~Ini()
{
	iniparser_freedict(m_ini);
}

void Ini::dump(const char* path)
{
	FILE* ini_file_pointer = stderr;//fopen(path, "w");

	iniparser_dump_ini(m_ini, ini_file_pointer);

	for (uint8_t i = 3; i && fclose(ini_file_pointer) == EOF; i--)
		std::cerr << "UNABLE TO CLOSE INI FILE! (" << path << ')' << std::endl;
}
