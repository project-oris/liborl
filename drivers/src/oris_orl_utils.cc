#include <chrono>
#include <iostream>
#include <thread>
#include <locale>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>
#include "oris_orl_utils.h"
#include "oris_orl_log.h"



std::string get_option_value(
    std::map<std::string, std::string> &options,
    const std::string &key,
    bool required,
    const std::string &defaultValue)

{
    if (auto search = options.find(key); search != options.end())
    {
        return search->second;
    }
    else if (required)
        throw std::runtime_error("Error: Options must be defined : " + key);

    return defaultValue;
}

void to_lower_string(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), (int (*)(int))std::tolower);
}