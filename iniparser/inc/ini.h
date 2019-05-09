#ifndef _INIPARSER_INI_H_
#define _INIPARSER_INI_H_

#include <iostream>
#include <fstream>
#include <iniparser.h>

class Ini
{
	private:
		dictionary* m_ini;

	public:
		Ini(const char* path);
		~Ini();

		void dump(const char* path);

		int         get_section_count     (void);
		std::string get_section_name      (int n);
		int         get_section_key_count (const std::string& s);
};

inline int Ini::get_section_count(void)
{
	return iniparser_getnsec(m_ini);
}

inline std::string Ini::get_section_name(int n)
{
	return iniparser_getsecname(m_ini, n);
}

inline int Ini::get_section_key_count(const std::string& s)
{
	return iniparser_getsecnkeys(m_ini, s.c_str());
}

#endif
