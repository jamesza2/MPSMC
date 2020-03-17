#ifndef time_summer_upper
#define time_summer_upper

#include <vector>
#include <string>
#include <map>
#include <ctime>

//Goal: Create a timer object that can store the total amount of time different processes take, as well as how many times they were run
//Can print either the total amount of time of a certain specified loop, or the total amount of time every process took.

class TimeEntry
{
	public:
		string key;
		int num_calls;
		double total_time;
		TimeEntry(){
			num_calls = 0;
			total_time = 0.;
			key = "";
		}
		TimeEntry(string k) : TimeEntry(){
			key = k;
		}
}
class TimerStorage{

}

#endif