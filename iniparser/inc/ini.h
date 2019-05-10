#ifndef _INIPARSER_INI_H_
#define _INIPARSER_INI_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <iniparser.h>
#include <alloca.h>

class Ini
{
	private:
		dictionary* m_ini;

	public:
		Ini(const char* path);
		~Ini();

		void dump(const char* path);

		int                      get_section_count    (void);
		std::string              get_section_name     (int n);
		int                      get_section_key_count(const std::string& section);
		std::vector<std::string> get_section_keys     (const std::string& section);
		std::string              get_string           (const std::string& section, const std::string& key);
};

inline int Ini::get_section_count(void)
{
	return iniparser_getnsec(m_ini);
}

inline std::string Ini::get_section_name(int n)
{
	return iniparser_getsecname(m_ini, n);
}

inline int Ini::get_section_key_count(const std::string& section)
{
	return iniparser_getsecnkeys(m_ini, section.c_str());
}

inline std::vector<std::string> Ini::get_section_keys(const std::string& section)
{
	int key_count = get_section_key_count(section);
	std::vector<std::string> keys(key_count);
	const char** tmp_keys = (const char**) alloca(key_count);

	iniparser_getseckeys(m_ini, section.c_str(), tmp_keys);

	for (int i = 0; i < key_count; i++) {
		keys[i] = tmp_keys[i];
		keys[i].erase(0, keys[i].find_first_of(':') + 1);
	}

	return keys;
}

inline std::string Ini::get_string(const std::string& section, const std::string& key)
{
	std::string ini_key = section + ':' + key;
	return iniparser_getstring(m_ini, ini_key.c_str(), "");
}

#endif
