//
//-$LICENSE
//

#ifndef _TRT_DRIVER_INTERNAL_H_
#define _TRT_DRIVER_INTERNAL_H_

#include <chrono>
#include <iostream>
#include <thread>
#include <locale>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <opencv2/opencv.hpp>
#include "oris_orl_types.h"
#include "oris_orl_config.h"
#include "trt_driver.h"
#include "NvInfer.h"

typedef enum TaskID
{
  DETECTION,
  SEGMENTATION,
  POSE,
  OBB, // Oritented BoundingBox
  CLASSIFICATION
} TaskID;

using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override;
};

struct Result
{
  float score;
  cv::Rect box;
  int class_id;
};

// enum class TRTPrecision
// {
//   FP32,
//   FP16,
//   INT8
// };

typedef struct _oris_orl_trt_driver
{
  std::string model_type;
  std::string model_file_type; // onnx or trt, etc...

  std::string task_type;

  // TRTPrecision precision;
  size_t input_height;
  size_t input_width;

  float normalization_value;
  bool m_normailization;
  
  std::string model_path;

  float m_ratio;
  // option valuie
  // Precision to use for GPU inference.
  // Precision precision = Precision::FP16;
  // std::string calibrationDataDirectoryPath;
  int32_t calibrationBatchSize = 128;
  // The batch size which should be optimized for.
  // int32_t optBatchSize = 1;
  // Maximum allowable batch size
  int32_t maxBatchSize = 16;

  // trt engine insances
  std::unique_ptr<nvinfer1::IBuilder> m_builder{nullptr}; // TensorRT builder used to create the engine
  std::unique_ptr<nvinfer1::IRuntime> m_runtime{nullptr}; // TensorRT runtime used to deserialize the engine
  std::unique_ptr<nvinfer1::ICudaEngine> m_engine{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext> m_context{nullptr};
  nvinfer1::IOptimizationProfile *m_profile = nullptr;

  // internal
  Logger m_logger;
  std::vector<void *> m_bindings;
  std::vector<size_t> m_binding_sizes;
  std::vector<nvinfer1::Dims> m_binding_dims;
  std::vector<nvinfer1::DataType> m_binding_types;
  std::vector<std::string> m_binding_names;
  std::map<std::string, void *> m_input_bindings;
  std::map<std::string, size_t> m_input_binding_size;
  std::map<std::string, void *> m_output_bindings;
  std::map<std::string, size_t> m_output_binding_size;
  std::map<std::string, nvinfer1::DataType> m_output_binding_type;
  std::map<std::string, std::vector<size_t>> m_input_dims;
  std::map<std::string, std::vector<size_t>> m_output_dims;

  int m_batch_size{1};
  int deviceIndex{0};
  int input_bindings{0};  // number of input binding
  int output_bindings{0}; // number of output binding
} _oris_orl_trt_driver;

void _check_cuda_error_code(cudaError_t code);

void _clear_cuda_buffers(void *driver);

float _img_preprocess(const cv::Mat &input, cv::Mat &converted, size_t height, size_t width, const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));

int __run_trt_infer(_oris_orl_trt_driver *_driver, std::vector<cv::cuda::GpuMat> &input, std::vector<float> &result);

int _detection_func(_oris_orl_trt_driver *_driver, cv::Mat &imgin, std::vector<oris::orl::Tensor> &result);
void _detection_func(_oris_orl_trt_driver *_driver, float *input, std::vector<oris::orl::Tensor> &result);

#endif //_TRT_DRIVER_INTERNAL_H_