#include <iostream>
#include <string>
#include <boost/program_options.hpp>

#define unix_style (boost::program_options::command_line_style::unix_style \
		  | boost::program_options::command_line_style::short_allow_next)

#define windows_style (boost::program_options::command_line_style::allow_long \
		     | boost::program_options::command_line_style::allow_short \
		     | boost::program_options::command_line_style::allow_slash_for_long \
		     | boost::program_options::command_line_style::allow_slash_for_short \
		     | boost::program_options::command_line_style::case_insensitive \
		     | boost::program_options::command_line_style::long_allow_next \
		     | boost::program_options::command_line_style::short_allow_next)

int main(int argc, char* argv[])
{
	boost::program_options::variables_map vmap;

	std::string interface_name;
	std::string bssid = "";
	uint8_t channel = 0;
	bool showack = false;

	boost::program_options::options_description desc("Options");
	desc.add_options()
		("bssid,b", boost::program_options::value<std::string>(),
		 "It will only show networks, matching the given bssid.")
		("channel,c", boost::program_options::value<uint8_t>()->default_value(3),
		 "Indicate the channel(s) to listen to. By default airodump-ng hops on all 2.4GHz channels.")
		("showack,",
		 "Prints ACK/CTS/RTS statistics. Helps in debugging and general injection optimization. It  is  indica‐"
		 "tion  if you inject, inject too fast, reach the AP, the frames are valid encrypted frames. Allows one"
		 "to detect \"hidden\" stations, which are too far away to capture high bitrate frames, as ACK frames are"
		 "sent at 1Mbps.");

	boost::program_options::options_description posparams("Positional parameters");
	posparams.add_options()
		("interface name", boost::program_options::value<std::string>(&interface_name)->required(), "");

	desc.add(posparams);

	boost::program_options::positional_options_description posopts;
	posopts.add("interface name", 1);

	try
	{
		boost::program_options::store(
				boost::program_options::command_line_parser(argc, argv)
				.options(desc)
				.positional(posopts)
				.style(unix_style)
				.run(), vmap);

		boost::program_options::notify(vmap);

		if (argc == 1 || vmap.count("help"))
		{
			std::cout << "Usage: " << argv[0] << std::endl << desc << std::endl;
			exit(0);
		}
	}
	catch (boost::program_options::error& poe)
	{
		std::cerr << poe.what() << std::endl << "Usage: " << argv[0] << std::endl << desc << std::endl;
		exit(EXIT_FAILURE);
	}

	if (vmap.count("bssid"))
		bssid = vmap["bssid"].as<std::string>();

	if (vmap.count("channel"))
		channel = vmap["channel"].as<uint8_t>();

	showack = vmap.count("showack") > 0;

	return 0;
}
