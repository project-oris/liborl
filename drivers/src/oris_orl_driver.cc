#include <dlfcn.h> // POSIX (Linux, macOS)
#include "oris_orl_driver.h"
#include "oris_orl_log.h"
#ifdef _WIN32
#include <windows.h>
typedef HINSTANCE LibraryHandle;
#define LOAD_LIBRARY(lib) LoadLibraryA(lib)
#define GET_SYMBOL(handle, name) GetProcAddress(handle, name)
#define UNLOAD_LIBRARY(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
typedef void *LibraryHandle;
#define LOAD_LIBRARY(lib) dlopen(lib, RTLD_LAZY)
#define GET_SYMBOL(handle, name) dlsym(handle, name)
#define UNLOAD_LIBRARY(handle) dlclose(handle)
#endif

using namespace oris::orl;

namespace oris
{
        namespace orl
        {

                bool _load_driver_library(driver_t &driver, const std::string &libraryPath)
                {

                        log_info("LOADING {}", libraryPath);
                        driver.libraryHandle = LOAD_LIBRARY(libraryPath.c_str());

                        if (!driver.libraryHandle)
                        {

#ifdef _WIN32
                                log_error("Error loading library: {}", GetLastError());
#else
                                log_error("Error loading library: {}", dlerror());
#endif
                                return false;
                        }

                        typedef void *(*InitEngineFunc)(config &_config);
                        driver._init_engine = (InitEngineFunc)GET_SYMBOL(driver.libraryHandle, "init_engine");

                        if (!driver._init_engine)
                        {
#ifdef _WIN32
                                log_error("Error finding symbol 'init_engine':  {}", GetLastError());
#else
                                log_error("Error finding symbol 'init_engine':  {}", dlerror());
#endif
                                UNLOAD_LIBRARY(driver.libraryHandle);

                                return false;
                        }

                        typedef int (*RunInferFunc)(void *driver, const std::vector<Tensor> &input,
                                                    std::vector<Tensor> &result, const options &options);
                        driver._run_infer = (RunInferFunc)GET_SYMBOL(driver.libraryHandle, "run_infer");
                        if (!driver._run_infer)
                        {
#ifdef _WIN32
                                log_error("Error finding symbol 'run_infer':  {}", GetLastError());
#else
                                log_error("Error finding symbol 'run_infer':  {}", dlerror());
#endif
                                UNLOAD_LIBRARY(driver.libraryHandle);

                                return false;
                        }

                        typedef void (*CloseEngineFunc)(void *driver);
                        driver._close_engine = (CloseEngineFunc)GET_SYMBOL(driver.libraryHandle, "close_engine");

                        if (!driver._close_engine)
                        {
#ifdef _WIN32
                                log_error("Error finding symbol 'close_engine':  {}", GetLastError());
#else
                                log_error("Error finding symbol 'close_engine':  {}", dlerror());
#endif
                                UNLOAD_LIBRARY(driver.libraryHandle);

                                return false;
                        }

                        typedef int (*GetInputShapeFunc)(void *driver, std::map<std::string, std::vector<size_t>> &shape);
                        driver._get_input_shape = (GetInputShapeFunc)GET_SYMBOL(driver.libraryHandle, "get_input_shape");

                        if (!driver._get_input_shape)
                        {
#ifdef _WIN32
                                log_error("Error finding symbol 'get_input_shapes':  {}", GetLastError());
#else
                                log_error("Error finding symbol 'get_input_shapes':  {}", dlerror());
#endif
                                UNLOAD_LIBRARY(driver.libraryHandle);

                                return false;
                        }

                        typedef int (*GetOutputShapeFunc)(void *driver, std::map<std::string, std::vector<size_t>> &shape);
                        driver._get_output_shape = (GetOutputShapeFunc)GET_SYMBOL(driver.libraryHandle, "get_output_shape");

                        if (!driver._get_output_shape)
                        {
#ifdef _WIN32
                                log_error("Error finding symbol 'get_output_shapes':  {}", GetLastError());
#else
                                log_error("Error finding symbol 'get_output_shapes':  {}", dlerror());
#endif
                                UNLOAD_LIBRARY(driver.libraryHandle);

                                return false;
                        }

                        return true;
                }

                bool load_driver(driver_t &_driver, const std::string &_filepath)
                {
                        if (!load_config(_driver.m_config, _filepath))
                        {
                                log_error("cannot find config file: {}", _filepath);
                                return false;
                        }

                        if (!_load_driver_library(_driver, _driver.m_config.m_driver_path))
                        {
                                log_error("load driver error");
                                return false;
                        }

                        _driver.engine_handle = _driver._init_engine(_driver.m_config);
                        if (_driver.engine_handle == nullptr)
                        {
                                log_error("AI engine initialization error");
                                unload_driver(_driver);
                        }

                        return true;
                }

                void unload_driver(driver_t &driver)
                {
                        UNLOAD_LIBRARY(driver.libraryHandle);
                }

                void close_engine(driver_t &driver)
                {
                        driver._close_engine(driver.engine_handle);
                }

                int get_input_shape(driver_t &driver, std::map<std::string, std::vector<size_t>> &shape)
                {
                        return driver._get_input_shape(driver.engine_handle, shape);
                }
                int get_output_shape(driver_t &driver, std::map<std::string, std::vector<size_t>> &shape)
                {
                        return driver._get_output_shape(driver.engine_handle, shape);
                }
                int run_infer(driver_t &driver, const std::vector<Tensor> &input,
                              std::vector<Tensor> &result, const options &options)
                {
                        return driver._run_infer(driver.engine_handle, input, result, options);
                }

        } // namespace orl

} // namespace oris