//
// $LICENSE
//

#ifndef _ORIS_ORL_DRIVER_H_
#define _ORIS_ORL_DRIVER_H_
#include <opencv2/opencv.hpp>
#include "oris_orl_types.h"
#include "oris_orl_config.h"
#include <map>
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif

    ////////////////////////////////////////////////////

    namespace oris
    {
        namespace orl
        {

#ifdef _WIN32
#include <windows.h>
            typedef HINSTANCE LibraryHandle;
#else
#include <dlfcn.h>
        typedef void *LibraryHandle;
#endif

            typedef struct driver_t
            {
                LibraryHandle libraryHandle;
                void *(*_init_engine)(config &_config);
                void (*_close_engine)(void *driver);
                int (*_get_input_shape)(void *driver, std::map<std::string, std::vector<size_t>> &shape);
                int (*_get_output_shape)(void *driver, std::map<std::string, std::vector<size_t>> &shape);
                int (*_run_infer)(void *driver, const std::vector<Tensor> &input,
                                  std::vector<Tensor> &result, const options &options);
                void *engine_handle;
                config m_config;
            } driver_t;

            bool load_driver(driver_t &driver, const std::string &config_path);
            void unload_driver(driver_t &driver);
            void close_engine(driver_t &driver);
            int get_input_shape(driver_t &driver, std::map<std::string, std::vector<size_t>> &shape);
            int get_output_shape(driver_t &driver, std::map<std::string, std::vector<size_t>> &shape);
            int run_infer(driver_t &driver, const std::vector<Tensor> &input,
                          std::vector<Tensor> &result, const options &options = null_option);

        } // namespace orl

    } // namespace oris_ai_rt

#ifdef __cplusplus
}
#endif
#endif // _ORIS_ORL_DRIVER_H_