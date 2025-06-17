#include <opencv2/opencv.hpp>
#include "oris_orl_types.h"

#include <chrono>

using namespace oris::orl;

namespace oris
{
  namespace orl
  {

    void hwc_to_chw(cv::InputArray src, cv::OutputArray dst) // hwc to flat
    {
      std::vector<cv::Mat> channels;
      cv::split(src, channels);

      // Stretch one-channel images to vector
      for (auto &img : channels)
      {
        img = img.reshape(1, 1);
      }
      
      // Concatenate three vectors to one
      cv::hconcat(channels, dst);
    }

    void cvMatToFloatTensor(const cv::Mat &img, Tensor &result, bool tochw, float normalization) // convert (HWC) to (CHW)
    {
      std::vector<size_t> dims;
      size_t channels = img.channels();
      size_t height = img.rows;
      size_t width = img.cols;

      if (normalization == 0)
        normalization = 1.0;

      if (tochw)
      {
        dims = {channels, height, width};
        result = makeTensor(DataType::FLOAT32, dims);
        result.format = "CHW";
        auto &dataptr = dynamic_cast<vector_t<float> *>(result.m_data.get())->get();
        for (size_t c = 0; c < channels; c++)
        {
          for (size_t j = 0, HW = height * width; j < HW; ++j)
          {
            dataptr[c * HW + j] = static_cast<float>(img.data[j * channels + 2 - c]) / normalization;
          }
        }
      }
      else
      {
        dims = {height, width, channels};
        result = makeTensor(DataType::FLOAT32, dims);
        result.format = "HWC";
        auto &dataptr = dynamic_cast<vector_t<float> *>(result.m_data.get())->get();
        for (size_t c = 0; c < channels; c++)
        {
          for (size_t j = 0, HW = height * width; j < HW; ++j)
          {
            dataptr[c * HW + j] = static_cast<float>(img.data[c * HW + j]) / normalization;
          }
        }
      }
    }

    void cvMatToTensor(const cv::Mat &img, Tensor &result, bool tochw) // convert (HWC) to (CHW)
    {
      if (img.empty())
        return;

      cv::Mat converted;
      if (tochw)
      {
        result.format = "CHW";
        result.shape = {static_cast<size_t>(img.channels()), static_cast<size_t>(img.rows), static_cast<size_t>(img.cols)};
        hwc_to_chw(img, converted);
      }
      else
      {
        result.format = "HWC";
        result.shape = {static_cast<size_t>(img.rows), static_cast<size_t>(img.cols), static_cast<size_t>(img.channels())};
        if (!img.isContinuous())
          converted = img.clone();
      }

      // Data Type 설정
      switch (img.type() & (unsigned int)(~CV_8UC4))
      {
      case CV_8U:
      case CV_8S:
        result.data_type = DataType::UINT8;
        if (tochw)
          result.m_data = std::make_shared<vector_t<uint8_t>>(
              converted.ptr<uint8_t>(), converted.ptr<uint8_t>() + converted.cols);
        else if (img.isContinuous())
          result.m_data = std::make_shared<vector_t<uint8_t>>(
              img.ptr<uint8_t>(), img.ptr<uint8_t>() + img.total() * img.channels());
        else
          result.m_data = std::make_shared<vector_t<uint8_t>>(
              converted.ptr<uint8_t>(), converted.ptr<uint8_t>() + converted.total() * converted.channels());
        break;
      case CV_16U:
      case CV_16S:
        result.data_type = DataType::INT16;
        if (tochw)
          result.m_data = std::make_shared<vector_t<int16_t>>(
              converted.ptr<int16_t>(), converted.ptr<int16_t>() + converted.cols);
        else if (img.isContinuous())
          result.m_data = std::make_shared<vector_t<int16_t>>(
              img.ptr<int16_t>(), img.ptr<int16_t>() + img.total() * img.channels());
        else
          result.m_data = std::make_shared<vector_t<int16_t>>(
              converted.ptr<int16_t>(), converted.ptr<int16_t>() + converted.total() * converted.channels());
        break;
      case CV_32S:
        result.data_type = DataType::INT32;
        if (tochw)
          result.m_data = std::make_shared<vector_t<int32_t>>(
              converted.ptr<int32_t>(), converted.ptr<int32_t>() + converted.cols);
        else if (img.isContinuous())
          result.m_data = std::make_shared<vector_t<int32_t>>(
              img.ptr<int32_t>(), img.ptr<int32_t>() + img.total() * img.channels());
        else
          result.m_data = std::make_shared<vector_t<int32_t>>(
              converted.ptr<int32_t>(), converted.ptr<int32_t>() + converted.total() * converted.channels());
        break;
      case CV_32F:
        result.data_type = DataType::FLOAT32;
        if (tochw)
          result.m_data = std::make_shared<vector_t<float>>(
              converted.ptr<float>(), converted.ptr<float>() + converted.cols);
        else if (img.isContinuous())
          result.m_data = std::make_shared<vector_t<float>>(
              img.ptr<float>(), img.ptr<float>() + img.total() * img.channels());
        else
          result.m_data = std::make_shared<vector_t<float>>(
              converted.ptr<float>(), converted.ptr<float>() + converted.total() * converted.channels());
        break;
      case CV_64F:
        result.data_type = DataType::FLOAT64;
        if (tochw)
          result.m_data = std::make_shared<vector_t<double>>(
              converted.ptr<double>(), converted.ptr<double>() + converted.cols);
        else if (img.isContinuous())
          result.m_data = std::make_shared<vector_t<double>>(
              img.ptr<double>(), img.ptr<double>() + img.total() * img.channels());
        else
          result.m_data = std::make_shared<vector_t<double>>(
              converted.ptr<double>(), converted.ptr<double>() + converted.total() * converted.channels());
        break;
      // 다른 OpenCV 데이터 타입에 대한 처리 추가
      default:
        throw std::runtime_error("Unsupported cv::Mat data type for TensorData conversion.");
      }
    }

    template <typename T>
    void chw_to_hwc(const std::vector<T> &buffer, int c, int h, int w, cv::Mat &output)
    {
      if (buffer.size() != static_cast<size_t>(c * h * w))
      {
        throw std::runtime_error("size mismatch.");
      }

      int cv_depth = -1;
      if (std::is_same<T, uint8_t>::value)
      {
        cv_depth = CV_8U;
      }
      else if (std::is_same<T, int16_t>::value)
      {
        cv_depth = CV_16S;
      }
      else if (std::is_same<T, float>::value)
      {
        cv_depth = CV_32F;
      }
      else if (std::is_same<T, int32_t>::value)
      {
        cv_depth = CV_32S; // signed int
      }
      else if (std::is_same<T, double>::value)
      {
        cv_depth = CV_64F;
      }
      else
      {
        throw std::runtime_error("Error: Unsupported data type for conversion.");
      }

      std::vector<cv::Mat> planes;
      planes.reserve(c);

      for (int p = 0; p < c; ++p)
      {
        const T *data_ptr = buffer.data() + static_cast<long long>(p) * h * w;
        cv::Mat channel_plane(h, w, CV_MAKETYPE(cv_depth, 1), const_cast<T *>(data_ptr));

        planes.push_back(channel_plane);
      }

      cv::merge(planes, output);
    }

    void TensorToCvMat(const Tensor &input, cv::Mat &output)
    {
      if (input.shape.empty())
      {
        output = cv::Mat(); // 0 size element
        return;
      }

      if (input.shape.size() > 3 || input.shape.size() < 2)
      {
        throw std::runtime_error("Unsupported TensorData to cv::Mat conversion.");
      }

      size_t height = 1;
      size_t width = 1;
      size_t channels = 1;
      bool tohwc = false;

      if (!input.format.compare("CHW"))
      {
        channels = input.shape[0];
        height = input.shape[1];
        width = input.shape[2];
        tohwc = true;
      }
      else if (!input.format.compare("HWC"))
      {
        height = input.shape[0];
        width = input.shape[1];
        channels = input.shape[2];
      }
      else
      {
        throw std::runtime_error("Unsupported format " + input.format);
      }

      switch (input.data_type)
      {
      case DataType::UINT8:
      {
        auto wrapper = dynamic_cast<vector_t<uint8_t> *>(input.m_data.get());
        if (tohwc)
          chw_to_hwc(wrapper->get(), channels, height, width, output);
        else
        {
          output = cv::Mat(height, width, CV_MAKETYPE(CV_8U, channels), const_cast<uint8_t *>(wrapper->get().data()));
        }
        break;
      }
      case DataType::INT16:
      {
        auto wrapper = dynamic_cast<vector_t<int16_t> *>(input.m_data.get());
        if (tohwc)
          chw_to_hwc(wrapper->get(), channels, height, width, output);
        else
        {
          output = cv::Mat(height, width, CV_MAKETYPE(CV_16S, channels), const_cast<int16_t *>(wrapper->get().data()));
        }
        break;
      }
      case DataType::INT32:
      {
        auto wrapper = dynamic_cast<vector_t<int32_t> *>(input.m_data.get());
        if (tohwc)
          chw_to_hwc(wrapper->get(), channels, height, width, output);
        else
        {
          output = cv::Mat(height, width, CV_MAKETYPE(CV_32S, channels), const_cast<int32_t *>(wrapper->get().data()));
        }
        break;
      }
      case DataType::FLOAT32:
      {
        auto wrapper = dynamic_cast<vector_t<float> *>(input.m_data.get());
        if (tohwc)
          chw_to_hwc(wrapper->get(), channels, height, width, output);
        else
        {
          output = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channels), const_cast<float *>(wrapper->get().data()));
        }
        break;
      }
      case DataType::FLOAT64:
      {
        auto wrapper = dynamic_cast<vector_t<double> *>(input.m_data.get());

        if (tohwc)
          chw_to_hwc(wrapper->get(), channels, height, width, output);
        else
        {
          output = cv::Mat(height, width, CV_MAKETYPE(CV_64F, channels), const_cast<double *>(wrapper->get().data()));
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported ors::DataType");
      }
    }

  } //   namespace orl

} // namespace oris