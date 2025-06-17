#include <iostream>
#include <dlfcn.h> // POSIX (Linux, macOS)
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <thread>
#include <unistd.h>
#include "ai_rt_driver.h"
#include "ai_rt_types.h"
#include "ai_rt_config.h"
#include "ai_rt_cv_support.h"
#include "ai_rt_yolo_support.h"

oris::ai_rt_driver_t driver_internal;

int main()
{
    std::string config_file_name{"./config.json"};
    if (!oris::load_driver(driver_internal, config_file_name))
    {
        std::cerr << "cannot load driver " << config_file_name << std::endl;
        return -1;
    }

    oris::ai_rt_options run_options;

    cv::Mat img_bgr = cv::imread("./dogs.png");

    cv::Mat converted_img;

    float ratio = oris::yolo::preprocess(img_bgr, converted_img, 640, 640, cv::Scalar(0, 0, 0));

    oris::Tensor img_tensor;

    oris::cvMatToFloatTensor(converted_img, img_tensor, true, 255.0);

    std::vector<oris::Tensor> input_tensors;
    std::vector<oris::Tensor> result;
    oris::ai_rt_options param;

    input_tensors.push_back(img_tensor);
    driver_internal._run_infer(driver_internal.engine_handle, param, input_tensors, result);
    if (result.size() > 0)
    {
#if 0        
        oris::Tensor &resultItem = result[0];

        auto wrapper = dynamic_cast<oris::vector_t<float> *>(resultItem.m_data.get());
        auto &item_data = wrapper->get();
        int k = 0;
        for (auto iter = item_data.begin(); iter != item_data.end();)
        {
            if (k > 10)
                break;
            for (int j = 0; j < 6; j++, ++iter)
            {
                std::cerr << *iter << ",";
            }
            std::cerr << std::endl;
            k++;
        }
#else        
        oris::Tensor &resultItem = result[0];
        std::vector<oris::yolo::detection_result_t> d_result;
        oris::yolo::tensor_to_detection_result(resultItem, d_result);
        for (auto &item : d_result)
        {
            item.x1 = std::clamp((int)(item.x1 * ratio), 0, img_bgr.cols);
            item.y1 = std::clamp((int)(item.y1 * ratio), 0, img_bgr.rows);
            item.x2 = std::clamp((int)(item.x2 * ratio), 0, img_bgr.cols);
            item.y2 = std::clamp((int)(item.y2 * ratio), 0, img_bgr.rows);

            std::cout<<"(" << item.x1 <<","<< item.y1 << ")-("
                << item.x2 << "," << item.y2 << ") -- [" << item.confidence << " : "
                << item.class_id << "]" << std::endl;
        }
#endif        
    }
}