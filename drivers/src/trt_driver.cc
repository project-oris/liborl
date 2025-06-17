//
//-$LICENSE
//

#include <chrono>
#include <iostream>
#include <thread>
#include <locale>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include "spdlog/spdlog.h"
#include "oris_orl_types.h"
#include "oris_orl_utils.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "oris_orl_log.h"
#include "trt_driver.h"
#include "trt_driver_internal.h"

// #define DEV_DEBUG

//for elapsed
/*
std::chrono::high_resolution_clock::time_point _begin, _end;
std::chrono::duration<double, std::milli> _elapsed;
#define BEGIN _begin = std::chrono::high_resolution_clock::now();
#define ELAPSED(x) { _end =std::chrono::high_resolution_clock::now(); _elapsed = _end-_begin; std::cout << x << " elapsed "<<_elapsed.count()<<" ms" << std::endl; }
*/

using namespace oris::orl;

void Logger::log(Severity severity, const char *msg) noexcept
{
  if (severity <= Severity::kWARNING)
  {
    std::cout << msg << std::endl;
  }
}

// A helper function to calculate memory usage
size_t get_volume_size(const nvinfer1::Dims &dims)
{
  return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
}

size_t get_volume_byte_size(const nvinfer1::Dims &dims, const nvinfer1::DataType &binding_type)
{
  size_t elem_size = 0;

  switch (binding_type)
  {
  case nvinfer1::DataType::kINT32:
    elem_size = 4;
    break;
  case nvinfer1::DataType::kFLOAT:
    elem_size = 4;
    break;
  case nvinfer1::DataType::kHALF:
    elem_size = 2;
    break;
  case nvinfer1::DataType::kINT8:
    elem_size = 1;
    break;
  default:;
  }

  return get_volume_size(dims) * elem_size;
}

void _check_cuda_error_code(cudaError_t code)
{
  if (code != 0)
  {
    std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                         "), with message: " + cudaGetErrorString(code);
    std::cerr << errMsg << std::endl;
    throw std::runtime_error(errMsg);
  }
}

inline void *_trt_cuda_alloc(size_t memSize)
{
  void *deviceMem;
  _check_cuda_error_code(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr)
  {
    throw std::runtime_error("Out of memory");
  }
  return deviceMem;
}

static auto StreamDeleter = [](cudaStream_t *pStream)
{
  if (pStream)
  {
    cudaStreamDestroy(*pStream);
    delete pStream;
  }
};

std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
  {
    pStream.reset(nullptr);
  }

  return pStream;
}

void _clear_cuda_buffers(void *driver)
{
  if (driver == nullptr)
    return;

  _oris_orl_trt_driver *_driver = (_oris_orl_trt_driver *)driver;

  if (!_driver->m_bindings.empty())
  {
    const auto numInputs = _driver->m_bindings.size();
    for (int32_t outputBinding = numInputs; outputBinding < _driver->m_engine->getNbIOTensors(); ++outputBinding)
    {
      _check_cuda_error_code(cudaFree(_driver->m_bindings[outputBinding]));
    }
    _driver->m_bindings.clear();
  }
}

bool _trt_build_engine(_oris_orl_trt_driver *ctx, std::string &onnxpath)
{
  ctx->m_builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(ctx->m_logger));
  if (!ctx->m_builder)
  {
    log_debug("Unable to make builder");
    return false;
  }

  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(ctx->m_builder->createNetworkV2(explicitBatch));
  if (!network)
  {
    log_debug("Unable to make network");
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(ctx->m_builder->createBuilderConfig());
  if (!config)
  {
    log_debug("Unable to make builder network");
    return false;
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, ctx->m_logger));
  if (!parser)
  {
    log_debug("Unable to make onnx parser");
    return false;
  }

  auto loaded = parser->parseFromFile(onnxpath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
  if (!loaded)
  {
    log_debug("Unable to load onnx file");
    return false;
  }

  auto profileStream = makeCudaStream();
  if (!profileStream)
  {
    log_debug("Unable to make profile cuda stream");
    return false;
  }
  config->setProfileStream(*profileStream);

  std::unique_ptr<nvinfer1::IHostMemory> plan{ctx->m_builder->buildSerializedNetwork(*network, *config)};
  if (!plan)
  {
    log_debug("Unable to make onnx pland");
    return false;
  }

  ctx->m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(ctx->m_logger));
  if (!ctx->m_runtime)
  {
    log_debug("Unable to make infer runtime");
    return false;
  }

  ctx->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(ctx->m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!ctx->m_engine)
  {
    log_debug("Unable to make cuda engine");
    return false;
  }

  return true;
}

bool _trt_load_engine(_oris_orl_trt_driver *ctx, std::string &enginepath)
{
  // load model
  std::ifstream file(enginepath, std::ios::binary | std::ios::ate);
  if (file.fail())
  {
    log_debug("Unable to load engine file {}", enginepath);
    return false;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size == 0)
  {
    log_debug("engine file is empty {}", enginepath);
    return false;
  }

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size))
  {
    log_debug("Unable to read engine file {}", enginepath);
    return false;
  }

  // crate runtime
  ctx->m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(ctx->m_logger)};
  if (ctx->m_runtime == nullptr)
  {
    log_debug("Unable to make infer runtime");
    return false;
  }

  ctx->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(ctx->m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (ctx->m_engine == nullptr)
  {
    log_debug("Unable to make cuda engine runtime");
    return false;
  }

  return true;
}

bool _trt_save_engine(_oris_orl_trt_driver *ctx, std::string const &enginepath)
{
  std::ofstream engine_file(enginepath, std::ios::binary);
  if (!engine_file)
  {
    log_debug("Unable to create engine file {}", enginepath);
    return false;
  }

  std::unique_ptr<nvinfer1::IHostMemory> serializedEngine{ctx->m_engine->serialize()};
  if (serializedEngine == nullptr)
  {
    log_debug("Engine serialization failed {}", enginepath);
    return false;
  }

  engine_file.write(static_cast<char *>(serializedEngine->data()), serializedEngine->size());
  return !engine_file.fail();
}

std::vector<size_t> _dims_to_vec(nvinfer1::Dims &src)
{
  std::vector<size_t> dst;
  dst.reserve(src.nbDims);
  for (int i = 0; i < src.nbDims; ++i)
  {
    dst.push_back(src.d[i]);
  }
  return dst;
}

void *init_engine(config &_config)
{
  spdlog::set_level(spdlog::level::debug);

  _oris_orl_trt_driver *driver = nullptr;
  try
  {
    driver = new _oris_orl_trt_driver();

    driver->m_batch_size = 1; // default 1

    // model type
    driver->model_type = _config.m_model;
    to_lower_string(driver->model_type);
    if (driver->model_type.compare("yolov8") != 0)
    {
      throw std::runtime_error("Error: Not supported Model Type : " + driver->model_type);
    }

    // engine file type
    std::string engine_type = _config.m_engine;
    to_lower_string(engine_type);
    if (engine_type.compare("trt") != 0)
    {
      throw std::runtime_error("Error: engine driver mismatch with engine type : " + engine_type);
    }

    // model task
    driver->task_type = _config.m_task;
    to_lower_string(driver->task_type);
    if (driver->task_type.compare(AI_TASK_DETECTION) != 0)
    {
      throw std::runtime_error("Error: Not supported Task Type : " + driver->task_type);
    }

    // model path
    driver->model_path = _config.m_model_path;

    // model file type
    driver->model_file_type = _config.m_model_file_type;
    to_lower_string(driver->model_file_type);

    // check engine or onnx
    if (driver->model_file_type.compare("onnx") == 0 ||
        (std::filesystem::path(driver->model_path).extension().compare(".onnx") == 0))
    {
      driver->model_file_type = "onnx";
      if (!_trt_build_engine(driver, driver->model_path))
      {
        throw std::runtime_error("Error: unable to make engine from onnx file : " + driver->model_path);
      }

      std::string engine_path = std::filesystem::path(driver->model_path).replace_extension("engine");
      _trt_save_engine(driver, engine_path);
    }
    else if (driver->model_file_type.compare("trt") == 0)
    {
      if (!_trt_load_engine(driver, driver->model_path))
      {
        throw std::runtime_error("Error: unable to make engine from engine file : " + driver->model_path);
      }
    }
    else
    {
      throw std::runtime_error("Error: unknown model file type : " + driver->model_path);
    }

    /// tensor information
    driver->m_normailization = _config.m_options.get_bool_option(AI_CONFIG_TENSOR_NORMALIZATION, false);
    driver->normalization_value = _config.m_options.get_float_option(AI_CONFIG_TENSOR_NORMALIZATION_BASE, 1.0);
    if (driver->normalization_value == 0)
    {
      driver->normalization_value = 1.0;
    }

    driver->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(driver->m_engine->createExecutionContext());
    if (driver->m_context == nullptr)
    {
      log_debug("Unable to execution context");
      throw std::runtime_error("Unable to execution context");
    }

    // init CUDA buffers
    //_clear_cuda_buffers(driver);
    
    const auto total_bindings = driver->m_engine->getNbIOTensors();
    driver->m_bindings.resize(total_bindings);
    driver->m_binding_sizes.resize(total_bindings);
    driver->m_binding_names.resize(total_bindings);
    driver->m_binding_dims.resize(total_bindings);
    driver->m_binding_types.resize(total_bindings);

    for (auto i = 0; i < total_bindings; ++i)
    {
      const char *binding_name = driver->m_engine->getIOTensorName(i);
      nvinfer1::DataType binding_type = driver->m_engine->getTensorDataType(binding_name);
      nvinfer1::Dims binding_dims = driver->m_engine->getTensorShape(binding_name);
      int64_t total_size = get_volume_byte_size(binding_dims, binding_type);
      if (total_size == 0)
      {
        log_debug("volume size cannot be zero : {} {} ", std::string(binding_name), (int)binding_type);
        throw std::runtime_error("volume size cannot be zero");
      }

      driver->m_binding_sizes[i] = total_size;
      driver->m_binding_names[i] = binding_name;
      driver->m_binding_dims[i] = binding_dims;
      driver->m_binding_types[i] = binding_type;
      driver->m_bindings[i] = _trt_cuda_alloc(total_size);
      // for enqueueV3      
      driver->m_context->setTensorAddress(binding_name, driver->m_bindings[i]);
      //
      auto iomode = driver->m_engine->getTensorIOMode(binding_name);
      if (iomode == nvinfer1::TensorIOMode::kINPUT)
      {
        if (binding_type != nvinfer1::DataType::kFLOAT)
        {
          log_debug("Error, the implementation currently only supports float inputs.. but it has {}.{} ", std::string(binding_name), (int)binding_type);
          throw std::runtime_error("Error, the implementation currently only supports float inputs");
        }
        driver->m_input_bindings[binding_name] = driver->m_bindings[i];
        driver->m_input_binding_size[binding_name] = total_size;
        driver->m_input_dims[binding_name] = _dims_to_vec(binding_dims);
        driver->input_bindings++;
#ifdef DEV_DEBUG
        log_debug("Input binding -> {}", binding_name);
        std::cerr << "DIM:";
        for (int i = 0; i < binding_dims.nbDims; ++i)
        {
          std::cerr << binding_dims.d[i] << ",";
        }
        std::cerr << std::endl;
#endif
      }
      else
      {
        driver->m_output_bindings[binding_name] = driver->m_bindings[i];
        driver->m_output_binding_size[binding_name] = total_size;
        driver->m_output_dims[binding_name] = _dims_to_vec(binding_dims);
        driver->output_bindings++;
#ifdef DEV_DEBUG
        log_debug("Output binding -> {}", binding_name);
        for (int i = 0; i < binding_dims.nbDims; ++i)
        {
          std::cerr << binding_dims.d[i] << ",";
        }
        std::cerr << std::endl;
#endif
      }
    }
  }
  catch (std::runtime_error &ex)
  {
    std::cerr << ex.what() << std::endl;
    if (driver != nullptr)
    {
      delete driver;
    }
    driver = nullptr;
  }

  return driver;
}

int get_input_shape(void *driver, std::map<std::string, std::vector<size_t>> &shape)
{
  _oris_orl_trt_driver *_driver = (_oris_orl_trt_driver *)driver;
  shape = _driver->m_input_dims;
  return shape.size();
}

int get_output_shape(void *driver, std::map<std::string, std::vector<size_t>> &shape)
{
  _oris_orl_trt_driver *_driver = (_oris_orl_trt_driver *)driver;
  shape = _driver->m_output_dims;
  return shape.size();
}

inline void _trt_copy_to_device(const std::vector<float> &input, void *dst, size_t count, const cudaStream_t &stream)
{
#ifdef TRT_DRIVER_DEBUG
  std::cerr << "input:" << input.size() * sizeof(float) << "...." << count << std::endl;
#endif
  assert(input.size() * sizeof(float) == count);

  _check_cuda_error_code(cudaMemcpyAsync(dst, input.data(), count, cudaMemcpyHostToDevice, stream));
}

inline void _trt_copy_to_host(void *src, std::vector<float> &output, size_t count, const cudaStream_t &stream)
{
#ifdef TRT_DRIVER_DEBUG
  std::cerr << "output:" << output.size() * sizeof(float) << "...." << count << std::endl;
#endif
  assert(output.size() * sizeof(float) == count);
  _check_cuda_error_code(cudaMemcpyAsync(output.data(), src, count, cudaMemcpyDeviceToHost, stream));
}

void _trt_run_infer(_oris_orl_trt_driver *driver, const std::vector<Tensor> &input, std::vector<Tensor> &result)
{
  if (input.size() > 1)
  {
    throw std::runtime_error("Error, Multiple input is not yet supported.!");
  }

  // if (driver->m_output_bindings.size() > 1) // support multiple
  //{
  //    throw std::runtime_error("Error, Multiple output is not yet supported.!");
  //  }

  cudaStream_t inferenceCudaStream;
  _check_cuda_error_code(cudaStreamCreate(&inferenceCudaStream));

  std::string input_name = driver->m_input_bindings.begin()->first;

  // std::cerr << "CUDA INPUT SIZE iS" << driver->m_input_binding_size[input_name] << std::endl;

  // std::cerr << "GET VOLUME " << get_volume_size(driver->m_input_dims[input_name]) << std::endl;
  //            << get_volume_size(driver->m_input_dims[input_name]) << std::endl;

  size_t elem_size = get_volume_size(driver->m_input_dims[input_name]);
  std::vector<float> input_buffer(elem_size);
  float norm_value = driver->normalization_value;

  switch (input[0].data_type)
  {
  case DataType::FLOAT32:
  {
    auto wrapper = dynamic_cast<vector_t<float> *>(input[0].m_data.get());
    auto data_ptr = wrapper->get().data();
    if (driver->m_normailization)
    {
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [norm_value](float val)
                     { return val / norm_value; });
    }
    else
      std::copy(data_ptr, data_ptr + elem_size, input_buffer.begin());
  }
  break;

  case DataType::UINT8:
  {
    auto wrapper = dynamic_cast<vector_t<uint8_t> *>(input[0].m_data.get());
    auto data_ptr = wrapper->get().data();
    if (driver->m_normailization)
    {
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [norm_value](uint8_t val)
                     { return val / norm_value; });
    }
    else
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [](uint8_t val)
                     { return static_cast<float>(val); });
  }
  break;
  case DataType::INT16:
  {
    auto wrapper = dynamic_cast<vector_t<int16_t> *>(input[0].m_data.get());
    auto data_ptr = wrapper->get().data();
    if (driver->m_normailization)
    {
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [norm_value](int16_t val)
                     { return val / norm_value; });
    }
    else
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [](int16_t val)
                     { return static_cast<float>(val); });
  }
  break;
  case DataType::INT32:
  {
    auto wrapper = dynamic_cast<vector_t<int32_t> *>(input[0].m_data.get());
    auto data_ptr = wrapper->get().data();
    if (driver->m_normailization)
    {
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [norm_value](int32_t val)
                     { return val / norm_value; });
    }
    else
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [](int32_t val)
                     { return static_cast<float>(val); });
  }
  break;
  case DataType::INT64:
  {
    auto wrapper = dynamic_cast<vector_t<int64_t> *>(input[0].m_data.get());
    auto data_ptr = wrapper->get().data();
    if (driver->m_normailization)
    {
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [norm_value](int64_t val)
                     { return val / norm_value; });
    }
    else
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [](int64_t val)
                     { return static_cast<float>(val); });
  }
  break;
  case DataType::FLOAT64:
  {
    auto wrapper = dynamic_cast<vector_t<double> *>(input[0].m_data.get());
    auto data_ptr = wrapper->get().data();
    if (driver->m_normailization)
    {
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [norm_value](double val)
                     { return val / norm_value; });
    }
    else
      std::transform(data_ptr, data_ptr + elem_size, input_buffer.begin(),
                     [](double val)
                     { return static_cast<float>(val); });
  }
  break;
  }

  _trt_copy_to_device(input_buffer, driver->m_input_bindings[input_name],
                      driver->m_input_binding_size[input_name], inferenceCudaStream);

  //BEGIN;
  //bool status = driver->m_context->enqueueV2(driver->m_bindings.data(), inferenceCudaStream, nullptr);
  bool status = driver->m_context->enqueueV3(inferenceCudaStream);
  //ELAPSED("enque");

  if (!status)
  {
    std::cerr << "ERROR: TensorRT inference failed." << std::endl;
    return;
  }

  for (std::pair<std::string, void *> itr : driver->m_output_bindings)
  {
    if (driver->m_output_dims[itr.first][0] != 1)
    {
      std::cerr << "output dim is " << driver->m_output_dims[itr.first][0] << std::endl;
      throw std::runtime_error("Error, batch output is not yet supported.!");
    }
    auto &t_dims = driver->m_output_dims[itr.first];
    if (t_dims.empty())
    {
      throw std::runtime_error("Error, dims cannot be empty!");
    }
    else
    {
      std::vector<size_t> out_dims(t_dims.size() - 1);
      std::copy(t_dims.begin() + 1, t_dims.end(), out_dims.begin());
      Tensor resultItem;
#ifdef DEV_DEBUG
      std::cerr << "output :" << itr.first << " (";
      for (auto d_itr : out_dims)
        std::cerr << d_itr << ",";
      std::cerr << ")...";

      std::cerr << "Binding Size =" << driver->m_output_binding_size[itr.first] << std::endl;
#endif

      resultItem = makeTensor(DataType::FLOAT32, out_dims);
      auto wrapper = dynamic_cast<vector_t<float> *>(resultItem.m_data.get());
      std::vector<float> &output_buffer = wrapper->get();
      _trt_copy_to_host(itr.second, output_buffer,
                        driver->m_output_binding_size[itr.first], inferenceCudaStream);
      result.push_back(resultItem);
    }
  }
  cudaStreamSynchronize(inferenceCudaStream);

  //   std::string output_name = driver->m_output_bindings.begin()->first;

  //   if (driver->m_output_dims[output_name][0] != 1)
  //   {
  //     std::cerr << "output dim is " << driver->m_output_dims[output_name][0] << std::endl;
  //     throw std::runtime_error("Error, Multiple output is not yet supported.!");
  //   }

  //   auto &t_dims = driver->m_output_dims[output_name];
  //   if (t_dims.empty())
  //   {
  //     throw std::runtime_error("Error, dims cannot be empty!");
  //   }

  //   std::vector<size_t> out_dims(t_dims.size() - 1);
  //   std::copy(t_dims.begin() + 1, t_dims.end(), out_dims.begin());

  //   Tensor resultItem;

  //   resultItem = makeTensor(DataType::FLOAT32, out_dims);
  //   auto wrapper = dynamic_cast<vector_t<float> *>(resultItem.m_data.get());
  //   std::vector<float> &output_buffer = wrapper->get();

  //   _trt_copy_to_host(driver->m_output_bindings[output_name], output_buffer,
  //                     driver->m_output_binding_size[output_name], inferenceCudaStream);

  //   cudaStreamSynchronize(inferenceCudaStream);
  //   result.push_back(resultItem);

  // #ifdef DEV_DEBUG

  //   for (int i = 0, k = 0; i < 30; i++)
  //   {
  //     for (int j = 0; j < 6; j++, k++)
  //     {
  //       std::cerr << output_buffer[k] << ",";
  //     }
  //     std::cerr << std::endl;
  //   }
  // #endif
}

int run_infer(void *driver, const std::vector<Tensor> &input,
              std::vector<Tensor> &result, const options &options)
{
  _oris_orl_trt_driver *_driver = (_oris_orl_trt_driver *)driver;

  int ret = 0;
#ifdef DEV_DEBUG
  std::cerr << "Input size " << input.size() << std::endl;
  for (size_t i = 0; i < input.size(); ++i)
  {
    for (size_t j = 0; j < input[i].shape.size(); ++j)
    {
      std::cerr << input[i].shape[j] << " ";
    }
    std::cerr << std::endl;
  }
#endif

  _trt_run_infer(_driver, input, result);

  // cv::Mat converted_img;
  // if (TensorToCvMat(input[0], converted_img) >= 0)
  // {
  //   if (task_id == DETECTION)
  //   {
  //     ret = _detection_func(_driver, converted_img, result);
  //   }
  // }

  return ret;
}

void close_engine(void *driver)
{
  _oris_orl_trt_driver *_driver = (_oris_orl_trt_driver *)driver;
  delete _driver;
}
