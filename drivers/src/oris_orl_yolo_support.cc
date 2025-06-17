#include <opencv2/opencv.hpp>
#include "oris_orl_types.h"
#include "oris_orl_log.h"
#include "oris_orl_yolo_support.h"

using namespace oris::orl;

namespace oris
{
  namespace yolo
  {

    float preprocess(const cv::Mat &input, cv::Mat &converted, size_t height, size_t width, const cv::Scalar &bgcolor)
    {
      cv::Mat converted_img;
      cv::cvtColor(input, converted_img, cv::COLOR_BGR2RGB);

      float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
      int unpad_w = r * input.cols;
      int unpad_h = r * input.rows;
      cv::Mat re(unpad_h, unpad_w, CV_8UC3);
      cv::resize(converted_img, re, re.size());

      converted.create(height, width, CV_8UC3);
      converted.setTo(bgcolor);
      re.copyTo(converted(cv::Rect(0, 0, re.cols, re.rows)));

      float ratio = 1.f / std::min(width / static_cast<float>(input.cols), height / static_cast<float>(input.rows));
      return ratio;
    }

    /*
     yolo nms 출력은
       (N, {x1,y1,x2,y2,confidence, classid)) 형태

       Tensor의 shape은 (N,6) 임..
    */

    template <typename T>
    void _vector_to_detection_result_t(const oris::orl::Tensor &data, std::vector<detection_result_t> &ret)
    {
      auto wrapper = dynamic_cast<vector_t<T> *>(data.m_data.get());
      auto data_ptr = wrapper->get();

      for (auto itr = data_ptr.begin(); itr != data_ptr.end(); itr++)
      {
        detection_result_t _t;
        _t.x1 = (uint16_t)*itr;
        itr++;
        _t.y1 = (uint16_t)*itr;
        itr++;
        _t.x2 = (uint16_t)*itr;
        itr++;
        _t.y2 = (uint16_t)*itr;
        itr++;
        _t.confidence = (float)(*itr);
        itr++;
        _t.class_id = (uint16_t)*itr;
        ret.push_back(_t);
      }
    }

    void tensor_to_detection_result(const oris::orl::Tensor &data, std::vector<detection_result_t> &ret)
    {
      if (data.shape.size() != 2)
      {
        log_error("illegal yolo detection result... {}", data.shape.size());
        //        std::cerr << "unknown yolo detection result..." << data.shape.size() << std::endl;
        throw std::runtime_error("illegal yolo detection result.");
      }

      if (data.shape[1] != 6)
      {
        log_error("illegal yolo detection result. {}", data.shape[1]);
        throw std::runtime_error("illegal yolo detection result.");
      }

      ret.clear();
      // std::cerr << " result size is " << data.shape[0] << std::endl;
      try
      {
        ret.reserve(data.shape[0]);
      }
      catch (std::exception &ex)
      {
        log_error("[ret] memory reservation fail. ");
        throw std::runtime_error("memory reservation fail.");
      }

      switch (data.data_type)
      {
      case DataType::INT16:
        _vector_to_detection_result_t<int16_t>(data, ret);
        break;
      case DataType::INT32:
        _vector_to_detection_result_t<uint32_t>(data, ret);
        break;
      case DataType::INT64:
        _vector_to_detection_result_t<uint64_t>(data, ret);
        break;
      case DataType::FLOAT32:
        _vector_to_detection_result_t<float>(data, ret);
        break;
      case DataType::FLOAT64:
        _vector_to_detection_result_t<double>(data, ret);
        break;
      default:
        log_debug("illegal yolo detection result. cannot be uint8");
        throw std::runtime_error("illegal yolo detection result");
      }
    }

    template <typename T>
    void _vector_to_segmentation_result_t(const oris::orl::Tensor &data, std::vector<segmentation_result_t> &ret)
    {
      auto wrapper = dynamic_cast<vector_t<T> *>(data.m_data.get());
      auto data_ptr = wrapper->get();

      for (auto itr = data_ptr.begin(); itr != data_ptr.end();)
      {
        segmentation_result_t _t;
        _t.detected.x1 = (uint16_t)*itr;
        itr++;
        _t.detected.y1 = (uint16_t)*itr;
        itr++;
        _t.detected.x2 = (uint16_t)*itr;
        itr++;
        _t.detected.y2 = (uint16_t)*itr;
        itr++;
        _t.detected.confidence = (float)(*itr);
        itr++;
        _t.detected.class_id = (uint16_t)*itr;
        itr++;
        std::copy(itr, itr + 32, _t.mask_coeff);
        itr += 32;
        if (_t.detected.confidence > 0)
          ret.push_back(_t);
      }
    }

    template <typename T>
    void _vector_to_segmentation_proto(const oris::orl::Tensor &data, segmentation_proto_mask_t &ret)
    {
      auto wrapper = dynamic_cast<vector_t<T> *>(data.m_data.get());
      auto data_ptr = wrapper->get();

      std::copy(data_ptr.begin(), data_ptr.end(), &ret.mask_proto[0]);
    }

    void tensor_to_segmentation_result(const std::vector<oris::orl::Tensor> &data, std::vector<segmentation_result_t> &ret0, segmentation_proto_mask_t &ret1)
    {
      if (data.size() != 2 || data[0].shape.size() != 2 || data[1].shape.size() != 3)
      {
        log_error("illegal yolo segmentation result... ");
        throw std::runtime_error("illegal yolo segmentation result.");
      }

      if (data[0].shape[1] != 38)
      {
        log_error("illegal yolo segmentation result. {}", data[0].shape[1]);
        throw std::runtime_error("illegal yolo segmentation result.");
      }

      if (data[1].shape[0] != 32 || data[1].shape[1] != 160 || data[1].shape[2] != 160)
      {
        log_error("illegal yolo segmentation result. ({},{},{})", data[1].shape[0], data[1].shape[1], data[1].shape[2]);
        throw std::runtime_error("illegal yolo segmentation result.");
      }

      ret0.clear();
      try
      {
        ret0.reserve(data[0].shape[0]);
      }
      catch (std::exception &ex)
      {
        log_error("[ret] memory reservation fail. ");
        throw std::runtime_error("memory reservation fail.");
      }

      switch (data[0].data_type)
      {
      case DataType::INT16:
        _vector_to_segmentation_result_t<int16_t>(data[0], ret0);
        break;
      case DataType::INT32:
        _vector_to_segmentation_result_t<uint32_t>(data[0], ret0);
        break;
      case DataType::INT64:
        _vector_to_segmentation_result_t<uint64_t>(data[0], ret0);
        break;
      case DataType::FLOAT32:
        _vector_to_segmentation_result_t<float>(data[0], ret0);
        break;
      case DataType::FLOAT64:
        _vector_to_segmentation_result_t<double>(data[0], ret0);
        break;
      default:
        log_debug("illegal yolo segmentation result0. cannot be uint8");
        throw std::runtime_error("illegal yolo segmentation result0");
      }

      switch (data[1].data_type)
      {
      case DataType::INT16:
        _vector_to_segmentation_proto<int16_t>(data[1], ret1);
        break;
      case DataType::INT32:
        _vector_to_segmentation_proto<uint32_t>(data[1], ret1);
        break;
      case DataType::INT64:
        _vector_to_segmentation_proto<uint64_t>(data[1], ret1);
        break;
      case DataType::FLOAT32:
        _vector_to_segmentation_proto<float>(data[1], ret1);
        break;
      case DataType::FLOAT64:
        _vector_to_segmentation_proto<double>(data[1], ret1);
        break;
      default:
        log_debug("illegal yolo segmentation result1. cannot be uint8");
        throw std::runtime_error("illegal yolo segmentation result1");
      }
    }

    cv::Mat _generate_seg_mask(
        const cv::Mat &mask_proto,      // (32, 160x160) cv::Mat
        std::vector<float> mask_coeff,  // (32)
        int coef_sz,                    // 160
        int input_sz,                   // 640
        int x1, int y1, int x2, int y2, // bbox
        double mask_threshold = 0.5)
    {
      cv::Mat mask_coeff_mat(1, mask_coeff.size(), CV_32F, mask_coeff.data());
      cv::Mat mask_feature = mask_coeff_mat * mask_proto;
      cv::Mat instance_mask = mask_feature.reshape(1, coef_sz);

      // sigmoid mask
      cv::exp(-instance_mask, instance_mask);
      instance_mask = 1.0f / (1.0f + instance_mask);

      //std::cerr << " mask shape = " << instance_mask.size() << std::endl;
      // std::cerr<<" mask_proto = " << mask_proto << std::endl;

      // mask resise to infer input (640x640)
      cv::Mat re_mask;
      cv::resize(instance_mask, re_mask, cv::Size(input_sz, input_sz), 0, 0, cv::INTER_LINEAR);

      // Create an empty mask initialized to zeros
      cv::Mat binary_mask = cv::Mat::zeros(re_mask.size(), CV_8U);

      // Define the region of interest (ROI)
      cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
      roi = roi & cv::Rect(0, 0, re_mask.cols, re_mask.rows);

      if (!roi.empty())
      {
        // Apply threshold only to the ROI of re_mask
        cv::Mat roi_re_mask = re_mask(roi);
        cv::Mat roi_binary_mask;
        cv::threshold(roi_re_mask, roi_binary_mask, mask_threshold, 255, cv::THRESH_BINARY);

        // Copy the thresholded ROI back to the corresponding region in binary_mask
        roi_binary_mask.copyTo(binary_mask(roi));
      }
      binary_mask.convertTo(binary_mask, CV_8U);

      return binary_mask;
    }

    void seg_postprocess(cv::Mat &result_img, const std::vector<oris::orl::Tensor> &data,
                         const cv::Mat &input_img, const int org_w, const int org_h, const float ratio,
                         const std::vector<std::string> &classes)
    {
      auto &bnm_data = dynamic_cast<oris::orl::vector_t<float> *>(data[0].m_data.get())->get();
      auto &fm_data = dynamic_cast<oris::orl::vector_t<float> *>(data[1].m_data.get())->get();

      int coef_h = data[1].shape[1];    // 160
      int coef_w = data[1].shape[2];    // 160
      int num_proto = data[1].shape[0]; // 32
      int input_sz = input_img.rows;    // 640
      // size_t strides = static_cast<size_t>(coef_h) * coef_w;

      cv::Mat mask_proto(num_proto, coef_h * coef_w, CV_32F, fm_data.data());
      // cv::Mat overlay = input_img.clone();
      cv::Mat out_img; // = input_img.clone();
      cv::cvtColor(input_img, out_img, cv::COLOR_BGR2RGB);
      cv::Mat overlay = out_img.clone();
      int class_size = classes.size();
      std::vector<cv::Scalar> class_color;
      for (int i = 0; i < class_size; i++)
      {
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);        
        class_color.push_back(color);
      }

      //bool is_detected = false;
      float cx, cy, w, h, cnf;
      int class_id;

      for (auto itr0 = bnm_data.begin(); itr0 != bnm_data.end(); itr0 += 32)
      {
        cx = *itr0;
        itr0++;
        cy = *itr0;
        itr0++;
        w = *itr0;
        itr0++;
        h = *itr0;
        itr0++;
        cnf = *itr0;
        itr0++;
        class_id = (int16_t)(*itr0);
        itr0++;

        if (cnf > 0 && cnf < 1.0 && w * h != 0 && class_id >= 0 && class_id < class_size)
        {
          //is_detected = true;
          std::vector<float> mask_coeffs(num_proto);
          std::copy(&(*itr0), &(*itr0) + num_proto, mask_coeffs.begin());
          cv::Mat seg_mask = _generate_seg_mask(mask_proto, mask_coeffs, coef_h, input_sz, cx, cy, w, h, 0.5);

          std::vector<std::vector<cv::Point>> contours;
          std::vector<cv::Vec4i> hierarchy;
          cv::findContours(seg_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
          cv::Scalar color = class_color[class_id];

          for (size_t i = 0; i < contours.size(); i++)
          {
            cv::drawContours(overlay, contours, (int)i, color, cv::FILLED, cv::LINE_8, hierarchy, 0);
            //cv::drawContours(overlay, contours, (int)i, color, cv::LINE_4, cv::LINE_8, hierarchy, 0);
          }

          double alpha = 0.3; // Transparency factor
          cv::addWeighted(overlay, alpha, out_img, 1 - alpha, 0, out_img);
        }
      } // for (auto itr0

      result_img = cv::Mat(org_h, org_w, CV_8UC3);
      cv::Mat re(input_sz * ratio, input_sz * ratio, CV_8UC3);
      cv::resize(out_img, re, re.size());

      cv::Rect roi(0, 0, org_w, org_h);
      re(roi).copyTo(result_img);
    }

  } // namespace yolo
} // namespace oris