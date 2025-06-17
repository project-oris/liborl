//
// $LICENSE
//

#ifndef _ORIS_ORL_CV_SUPPORT_
#define _ORIS_ORL_CV_SUPPORT_
#include <string>
#include <opencv2/opencv.hpp>
#include "oris_orl_types.h"

namespace oris
{
  namespace orl
  {
    void cvMatToTensor(const cv::Mat &img, Tensor &result, bool tochw = false); // convert HWC -> CHW

    void cvMatToFloatTensor(const cv::Mat &img, Tensor &result, bool tochw = false, float normalization = 1.0); // convert (HWC) to (CHW)
    /*
      cvMatToFloatTensor:
        tochw : HWC to CHW
        normalization : value devided by normalization
    */

    void TensorToCvMat(const Tensor &tensor_data, cv::Mat &output_mat); // convert CHW -> HWC
  } // namespace orl
} // namespace oris

#endif // _ORIS_ORL_CV_SUPPORT_