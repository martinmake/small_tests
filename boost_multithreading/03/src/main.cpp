#include <iostream>
#include <sstream>
#include <vector>
#include <map>

#include <boost/thread.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/move/move.hpp>

void do_work(const std::string& name)
{
	std::stringstream sout;

	sout << "[NAME=" << name << "]" << "[ID=" << boost::this_thread::get_id() << "]" << " DOING WORK" << std::endl;
	std::cout << sout.str();

	boost::this_thread::sleep_for(boost::chrono::seconds(2));
}

int main(void)
{
	using threadmap = std::map<boost::thread::id, boost::thread>;

	threadmap tmap;

	std::vector<std::string> tnames =
	{
		"THREAD_0", "THREAD_1", "THREAD_2", "THREAD_3",
		"THREAD_4", "THREAD_5", "THREAD_6", "THREAD_7",
		"THREAD_8", "THREAD_9", "THREAD_A", "THREAD_B",
		"THREAD_C", "THREAD_D", "THREAD_E", "THREAD_F"
	};

	for (const std::string& name : tnames)
	{
		boost::thread t(do_work, name);
		tmap[t.get_id()] = boost::move(t);
	}

	for (std::pair<const boost::thread::id, boost::thread>& thread_entry : tmap)
	{
		thread_entry.second.join();
		std::cout << thread_entry.first << " RETURNED" << std::endl;
	}


	return 0;
}
