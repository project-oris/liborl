// 
//-$LICENSE
// 


#ifndef _ORIS_ORL_LOG_H_
#define _ORIS_ORL_LOG_H_

#define ORIS_LOG_ENABLED

#ifdef ORIS_LOG_ENABLED
#include <spdlog/spdlog.h>
#include <string>

class TraceLog {
    std::string m_file;
    std::string m_func;
    int m_line;
public:
    template <typename... Args>
    TraceLog(const std::string &file, const std::string &func, int line, const std::string &msg = "", Args... args){
        m_file=file, m_func=func, m_line=line;
        spdlog::info("[T-IN : {}:{}]\n...\t{}..\n\t" + msg, func, line, file, args...); 
    }
    virtual ~TraceLog() {
        spdlog::info("[T-OUT: {}:{}]",m_func, m_line);
    }
};

#define log_trace(...) TraceLog _m_log(__FILE__, __func__, __LINE__, ##__VA_ARGS__)
#define log_info(fmt, ...) spdlog::info(fmt, ##__VA_ARGS__)
#define log_debug(fmt, ...) spdlog::debug("[{}:{}]" fmt,__func__,__LINE__, ##__VA_ARGS__)
#define log_warn(fmt, ...) spdlog::warn(fmt, ##__VA_ARGS__)
#define log_error(fmt, ...) spdlog::error(fmt, ##__VA_ARGS__)
#define log_critical(fmt, ...) spdlog::critical(fmt, ##__VA_ARGS__)
#else
#define log_enter(...)
#define log_info(...) 
#define log_debug(...) 
#endif

#endif // _ORIS_ORL_LOG_H_