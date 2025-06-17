//
// $LICENSE
//

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <locale>
#include "oris_orl_types.h"
#include "oris_orl_config.h"
#include "oris_orl_cv_support.h"
#include "oris_orl_yolo_support.h"
#include "oris_ai/model/model.h"

// #define ORIS_AI_DEBUG

void warmup(std::unique_ptr<oris_ai::Model> &model, size_t detect_height, size_t detect_width)
{
  // Create dummy data with detect_height x detect_width x 3
  cv::Mat dummy_image(detect_height, detect_width, CV_8UC3, cv::Scalar(0, 0, 0));
  // Run inference
  model->SetInputImageTensor(dummy_image);
  model->Forward();
  model->PostProcess(0.7f); // default 0.25
}

extern "C"
{
  typedef struct _oris_ai_driver
  {
    oris_ai::ModelType model_type;
    oris_ai::TaskType task_type;
    oris_ai::Device device;
    std::unique_ptr<oris_ai::Model> model;
    size_t input_height;
    size_t input_width;
    float normalization_value;
    std::string model_path;
    bool m_nms;
    bool m_tochw;
    bool m_normailization;
  } oris_ai_driver;

  typedef enum TaskID
  {
    DETECTION,
    CLASSIFICATION
  } TaskID;

  void *init_engine(oris::orl::config &_config)
  {
    oris_ai_driver *driver = new oris_ai_driver();

    if (_config.m_model.compare("yolov8n") == 0)
    {
      driver->model_type = oris_ai::ModelType::YOLOv8n;
    }
    else if (_config.m_model.compare("mobilenetyolov8n") == 0)
    {
      driver->model_type = oris_ai::ModelType::MobileNetYOLOv8n;
    }
    else
    {
      throw std::runtime_error("Error: Not supported Model :" + _config.m_model);
      std::cerr << "Error: Not supported Model Type : " << _config.m_model << std::endl;
      return nullptr;
    }

    if (_config.m_task.compare("detection") == 0)
    {
      driver->task_type = oris_ai::TaskType::Detection;
    }
    else if (_config.m_task.compare("segmentation") == 0)
    {
      driver->task_type = oris_ai::TaskType::Segmentation;
    }
    else
    {
      throw std::runtime_error("Error: Not supported Task :" + _config.m_task);
      std::cerr << "Error: Not supported Task Type : " << _config.m_task << std::endl;
      return nullptr;
    }

    if (_config.m_device.compare("cpu") == 0)
    {
      driver->device = oris_ai::Device::CPU;
    }
    else if (_config.m_device.compare("gpu") == 0)
    {
      driver->device = oris_ai::Device::GPU;
    }
    else
    {
      throw std::runtime_error("Error: Not supported device :" + _config.m_device);
      std::cerr << "Error: Not supported device : " << _config.m_device << std::endl;
      return nullptr;
    }

    driver->m_normailization = _config.m_options.get_bool_option(AI_CONFIG_TENSOR_NORMALIZATION, false);
    driver->normalization_value = _config.m_options.get_float_option(AI_CONFIG_TENSOR_NORMALIZATION_BASE, 255.0);
    std::string shape = _config.m_options.get_str_option(AI_CONFIG_TENSOR_INPUT_SHAPE, "(384,640)");

    std::vector<int> shape_val = oris::orl::parse_shape(shape);
    if (shape_val.size() != 2)
    {
      throw std::runtime_error("Error: illegal tensor input shape :" + shape);
    }
    driver->input_height = shape_val[0];
    driver->input_width = shape_val[1];
    driver->m_tochw = _config.m_options.get_bool_option(AI_CONFIG_YOLO_TOCHW, false);
    driver->m_nms = _config.m_options.get_bool_option(AI_CONFIG_YOLO_NMS, false);
    driver->model_path = _config.m_model_path;

#ifdef ORIS_AI_DEBUG

    std::cerr << " CHECK !!!!!!!! " << driver->input_width << "," << driver->input_height << std::endl;
    std::cerr << " NMS!! " << driver->m_nms << shape << std::endl;
    std::cerr << " m_tochw!! " << driver->m_tochw << std::endl;

#endif // ORIS_AI_DEBUG

    driver->model = oris_ai::CreateModel(driver->model_type, driver->task_type, driver->device);
    driver->model->CreateInputImageTensor(driver->input_height, driver->input_width, driver->normalization_value);
    driver->model->Open(driver->model_path);
    // warmup(driver->model, driver->input_height, driver->input_width);

    return driver;
  }

  int _detection_func(oris_ai_driver *_driver, cv::Mat &input, std::vector<oris::orl::Tensor> &result)
  {
    _driver->model->SetInputImageTensor(input);
    _driver->model->Forward();
    _driver->model->PostProcess(0.7f);

    std::vector<oris_ai::Detection> detections;

    detections = _driver->model->GetDetectionResults();

    if (detections.empty())
    {
      return 0;
    }

    oris::orl::Tensor resultItem;

    resultItem.shape = {static_cast<size_t>(detections.size()), 6};
    resultItem.data_type = oris::orl::DataType::FLOAT32;
    resultItem.m_data = std::make_shared<oris::orl::vector_t<float>>();
    auto wrapper = dynamic_cast<oris::orl::vector_t<float> *>(resultItem.m_data.get());
    wrapper->get().reserve(detections.size() + 6);
    for (size_t i = 0; i < detections.size(); ++i)
    {
      wrapper->get().push_back(static_cast<float>(detections[i].x1));
      wrapper->get().push_back(static_cast<float>(detections[i].y1));
      wrapper->get().push_back(static_cast<float>(detections[i].x2));
      wrapper->get().push_back(static_cast<float>(detections[i].y2));
      wrapper->get().push_back(detections[i].confidence);
      wrapper->get().push_back(static_cast<float>(detections[i].class_id));
    }

    result.push_back(resultItem);

    return result.size();
  }

  int get_input_shape(void *driver, std::map<std::string, std::vector<int>> &shape)
  {
    oris_ai_driver *_driver = (oris_ai_driver *)driver;

    std::vector<int> shape_value = {1, 3, (int)_driver->input_height, (int)_driver->input_width};

    shape["images"] = shape_value;

    return 1;
  }

  int get_output_shape(void *driver, std::map<std::string, std::vector<int>> &shape)
  {
    oris_ai_driver *_driver = (oris_ai_driver *)driver;

    if (_driver->task_type == oris_ai::TaskType::Detection)
    {
      std::vector<int> shape_value = {1, 3, 300, 6}; // yolo output
      shape["output0"] = shape_value;
    }
    else if (_driver->task_type == oris_ai::TaskType::Segmentation)
    {
      std::vector<int> shape_value = {1, 300, 38}; // yolo output
      shape["output0"] = shape_value;
    }

    return 1;
  }

  int run_infer(void *driver, const std::vector<oris::orl::Tensor> &input,
                std::vector<oris::orl::Tensor> &result, const oris::orl::options &options)
  {
    oris_ai_driver *_driver = (oris_ai_driver *)driver;

    cv::Mat converted_img;

    int ret = 0;

    TensorToCvMat(input[0], converted_img);

    {
      if (_driver->task_type == oris_ai::TaskType::Detection)
      {
        ret = _detection_func(_driver, converted_img, result);

#ifdef DRIVER_DEBUG
        std::vector<std::string> classes{"person", "car", "ev charging station", "cctv", "gas station"};

        std::vector<oris_oris_orl::_detection_result_t> detections;

        if (ret != 0)
        {
          std::cout << "Detected Class ";
          size_t num_detections = result[0].shape[0];
          const float *data_ptr = static_cast<const float *>(result[0].data);
          detections.resize(num_detections);

          for (size_t i = 0; i < num_detections; ++i)
          {
            {
              detections[i].class_id = static_cast<uint16_t>(data_ptr[i * 6 + 0]);
              detections[i].confidence = data_ptr[i * 6 + 1];
              detections[i].x1 = static_cast<uint16_t>(data_ptr[i * 6 + 2]);
              detections[i].y1 = static_cast<uint16_t>(data_ptr[i * 6 + 3]);
              detections[i].x2 = static_cast<uint16_t>(data_ptr[i * 6 + 4]);
              detections[i].y2 = static_cast<uint16_t>(data_ptr[i * 6 + 5]);

              std::cout << classes[detections[i].class_id]
                        << " (" << detections[i].x1 << ","
                        << detections[i].y1
                        << ")-("
                        << detections[i].x2 << ","
                        << detections[i].y2 << ")" << std::endl;
            }
          }
        }
#endif
      }
    }

    return ret;
  }

  void close_engine(void *driver)
  {
    oris_ai_driver *_driver = (oris_ai_driver *)driver;
    delete _driver;
  }

} // extern "C"