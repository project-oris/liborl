//
// $LICENSE
//

#ifndef _ORIS_ORL_YOLO_SUPPORT_H_
#define _ORIS_ORL_YOLO_SUPPORT_H_

#include <stdint.h>
#include <vector>
#include <string>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include "oris_orl_types.h"

namespace oris
{
  namespace yolo
  {
    typedef struct detection_result_t
    {
      uint16_t class_id{0};
      float confidence{0.0f};
      uint16_t x1{0};
      uint16_t y1{0};
      uint16_t x2{0};
      uint16_t y2{0};
    } __attribute__((packed)) detection_result_t;  /// topic: yolo/detect/detected

    typedef struct segmentation_result_t
    {      
      detection_result_t detected;
      float mask_coeff[32]{0}; 
    } segmentation_result_t;  // topic: yolo/seg/detected

    typedef struct segmentation_proto_mask_t
    {
      float mask_proto[32 * 160 * 160]{0};
    } segmentation_proto_mask_t;   // topic: yolo/seg/protomask

    float preprocess(const cv::Mat &input, cv::Mat &converted, size_t height, size_t width, const cv::Scalar &bgcolor);
    void tensor_to_detection_result(const oris::orl::Tensor &data, std::vector<detection_result_t> &ret);
    void tensor_to_segmentation_result(const oris::orl::Tensor &data, std::vector<segmentation_result_t> &ret, segmentation_proto_mask_t &proto);
    void seg_postprocess(cv::Mat &result_img, const std::vector<oris::orl::Tensor> &data, const cv::Mat &input_img, 
        const int org_w, const int org_h, const float ratio, const std::vector<std::string> &classes);

  } // namespace yolo
} // namespace oris

#endif // _ORIS_ORL_OLO_SUPPORT_H_