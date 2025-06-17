//
// $LICENSE
//

#ifndef _ORIS_ORL_UTILS_H_
#define _ORIS_ORL_UTILS_H_
#include <map>
#include <string>

std::string get_option_value(std::map<std::string, std::string> &options,
    const std::string &key, bool required = true,const std::string &defaultValue = "");
void to_lower_string(std::string &str);
#endif // _ORIS_ORL_UTILS_H_