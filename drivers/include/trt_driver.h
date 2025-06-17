//
//-$LICENSE
//

#ifndef _TRT_DRIVER_H_
#define _TRT_DRIVER_H_
#include <vector>
#include "oris_orl_types.h"
#include "oris_orl_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

  void *init_engine(oris::orl::config &_config);
  void close_engine(void *driver);
  int get_input_shape(void *driver, std::map<std::string, std::vector<size_t>> &shape);
  int get_output_shape(void *driver, std::map<std::string, std::vector<size_t>> &shape);
  int run_infer(void *driver, const std::vector<oris::orl::Tensor> &input, 
    std::vector<oris::orl::Tensor> &result, const oris::orl::options &options= oris::orl::null_option);
#ifdef __cplusplus
} // extern "C"
#endif

#endif //_TRT_DRIVER_H_