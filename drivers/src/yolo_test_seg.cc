#include <iostream>
#include <dlfcn.h> // POSIX (Linux, macOS)
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <numeric> 
#include <random> // for random colors
#include "oris_orl_driver.h"
#include "oris_orl_types.h"
#include "oris_orl_config.h"
#include "oris_orl_cv_support.h"
#include "oris_orl_yolo_support.h"

oris::orl::driver_t driver_internal;

std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                 "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                 "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                 "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                                 "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                 "scissors", "teddy bear", "hair drier", "toothbrush"};

std::vector<cv::Scalar> class_color;

std::chrono::high_resolution_clock::time_point begin, end;
std::chrono::duration<double, std::milli> elapsed;

void generateColorTable(int classnum) {
    for (int i=0; i<classnum;i++){
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        class_color.push_back(color); 
    }        
}

cv::Mat generate_seg_mask(
    const cv::Mat &mask_proto,  // (32, 160x160) cv::Mat
    std::vector<float> mask_coeff, // (32)
    int coef_sz,  // 160
    int input_sz, // 640
    int x1, int y1,int x2, int y2,    // bbox
    double mask_threshold=0.5
)
{
    cv::Mat mask_coeff_mat(1, mask_coeff.size(), CV_32F, mask_coeff.data());
    cv::Mat mask_feature = mask_coeff_mat * mask_proto;
    cv::Mat instance_mask = mask_feature.reshape(1, coef_sz);
    
    // sigmoid mask
    cv::exp(-instance_mask, instance_mask);
    instance_mask = 1.0f / (1.0f + instance_mask);

    std::cerr<<" mask shape = " << instance_mask.size() << std::endl;    
    //std::cerr<<" mask_proto = " << mask_proto << std::endl;


    // mask resise to infer input (640x640)
    cv::Mat re_mask;
    cv::resize(instance_mask, re_mask, cv::Size(input_sz, input_sz), 0, 0, cv::INTER_LINEAR);    

    //? cv::Mat binary_mask;    
    //? cv::threshold(re_mask, binary_mask, mask_threshold, 255, cv::THRESH_BINARY);
    //? binary_mask.convertTo(binary_mask, CV_8U); // Convert to 8-bit unsigned for findContours

    

    // Create an empty mask initialized to zeros
    cv::Mat binary_mask = cv::Mat::zeros(re_mask.size(), CV_8U);

    // Define the region of interest (ROI)
    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    roi = roi & cv::Rect(0, 0, re_mask.cols, re_mask.rows);

    if (!roi.empty()) {
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

cv::Mat yolo_seg_masks_and_draw(
    oris::orl::Tensor &box_and_masks,
    oris::orl::Tensor &final_masks,        
    int input_w,
    int input_h,
    cv::Mat &input_img,
    cv::Mat &org_img,
    float ratio)
{

    auto w_bnm = dynamic_cast<oris::orl::vector_t<float> *>(box_and_masks.m_data.get());
    auto w_fm = dynamic_cast<oris::orl::vector_t<float> *>(final_masks.m_data.get());
    auto &bnm_data = w_bnm->get();
    auto &fm_data = w_fm->get();
    int k = 0;
    int class_id;
    float cx, cy, w, h, cnf;
    int i;

    //cv::Mat result_img = input_img.clone();    
    cv::Mat result_img;
    cv::cvtColor(input_img, result_img, cv::COLOR_BGR2RGB);
    int img_h = org_img.rows;
    int img_w = org_img.cols;
    int coef_h = 160;
    int coef_w = 160;
    int num_proto = 32;

    size_t strides = static_cast<size_t>(coef_h) * coef_w;    
    std::vector<cv::Mat> mask_coeff_vec;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255);

    std::cerr << "h:" << img_h << ",w" << img_w << std::endl;
    std::cerr << "ratio:" << ratio << std::endl;

    cv::Mat mask_proto(num_proto, coef_h*coef_w, CV_32F, fm_data.data());

    cv::Mat overlay; 
    cv::cvtColor(input_img, overlay, cv::COLOR_BGR2RGB);

    std::vector<int> bbox_class;
    std::vector<float> bbox_cnf;
    std::vector<cv::Rect> bbox;

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

        // itr0+=32; // mask coefficient
        if (cnf > 0 && cnf < 1.0 && w * h != 0)
        {
            std::cerr << "cx=" << cx << ",cy=" << cy;
            std::cerr << ",w=" << w << ",h=" << h << ",cnf=" << cnf << std::endl;

            

            if (class_id < 0 || class_id >= classes.size())
            {
                std::cerr << "Illegal Class ID" << std::endl;
                break;
            }        

            int x1 = std::clamp(static_cast<int>((cx) * ratio), 0, org_img.cols);
            int y1 = std::clamp(static_cast<int>((cy) * ratio), 0, org_img.rows);
            int x2 = std::clamp(static_cast<int>((w) * ratio), 0, org_img.cols);
            int y2 = std::clamp(static_cast<int>((h) * ratio), 0, org_img.rows); 

            bbox_cnf.push_back(cnf);
            bbox.push_back(cv::Rect(x1,y1,x2-x1, y2-y1));
            bbox_class.push_back(class_id);            
                        
            std::vector<float> mask_coeffs(num_proto);
            std::copy(&(*itr0), &(*itr0) + num_proto, mask_coeffs.begin());

            cv::Mat seg_mask = generate_seg_mask(mask_proto, mask_coeffs, 160, 640, cx, cy, w, h, 0.5);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(seg_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::cerr<<"Contours ..." << contours.size() << std::endl;
            std::cerr<<"ContoursPoints..." << contours[0].size() << std::endl;
            std::cerr<<"Class ..." << classes[class_id]<< std::endl;            

            cv::Scalar color = class_color[class_id];        
            
            for (size_t i = 0; i < contours.size(); i++) {                                                
                cv::drawContours(overlay, contours, (int)i, color, cv::FILLED, cv::LINE_8, hierarchy, 0);
            }            

            double alpha = 0.3; // Transparency factor
            cv::addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img);          
        }
    }

    if (bbox.size() != 0)
    {
        cv::Mat converted(org_img.rows, org_img.cols, CV_8UC3);
        cv::Mat re(640*ratio, 640*ratio,CV_8UC3);
        cv::resize(result_img, re, re.size());
        cv::Rect roi(0,0,org_img.cols,org_img.rows);
        re(roi).copyTo(converted);
        return converted;
    }
    

    return result_img;
}

int main()
{
    std::string config_file_name{"../assets/seg_config.json"};
    if (!oris::orl::load_driver(driver_internal, config_file_name))
    {
        std::cerr << "cannot load driver " << config_file_name << std::endl;
        return -1;
    }

    cv::Mat img_bgr = cv::imread("../assets/dogs.png");

    generateColorTable( classes.size());

    cv::Mat converted_img;


    begin = std::chrono::high_resolution_clock::now();
    float ratio = oris::yolo::preprocess(img_bgr, converted_img, 640, 640, cv::Scalar(0, 0, 0));
    end = std::chrono::high_resolution_clock::now();

    elapsed = end - begin;
    std::cout << "preprocess time: " << elapsed.count() << " ms" << std::endl;

    oris::orl::Tensor img_tensor;

    begin = std::chrono::high_resolution_clock::now();
    oris::orl::cvMatToTensor(converted_img, img_tensor, true);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - begin;
    std::cout << "Mato to Tensor time: " << elapsed.count() << " ms" << std::endl;

    std::vector<oris::orl::Tensor> input_tensors;
    std::vector<oris::orl::Tensor> result;

    input_tensors.push_back(img_tensor);

    begin = std::chrono::high_resolution_clock::now();
    oris::orl::run_infer(driver_internal, input_tensors, result);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - begin;
    std::cout << "infer time: " << elapsed.count() << " ms" << std::endl;

    std::cerr << "CLASS " << classes.size() << std::endl;

    if (result.size() == 2)
    {
        std::cerr << "result sz : " << result.size() << std::endl;
        oris::orl::Tensor &resultItem1 = result[0]; // nms_output_with_scaled_boxes_and_masks
        oris::orl::Tensor &resultItem2 = result[1]; // final_masks

        std::cerr << " Shape for 0 : (";
        for (auto d_itr : resultItem1.shape)
            std::cerr << d_itr << ",";
        std::cerr << ")" << std::endl;
        std::cerr << " Shape for 1 : (";
        for (auto d_itr : resultItem2.shape)
            std::cerr << d_itr << ",";
        std::cerr << ")" << std::endl;

        //cv::Mat resultImg = yolo_seg_masks_and_draw(resultItem1, resultItem2, 640, 640, converted_img, img_bgr, ratio);
        cv::Mat resultImg;
        begin = std::chrono::high_resolution_clock::now();
        oris::yolo::seg_postprocess(resultImg, result, converted_img, img_bgr.cols, img_bgr.rows, ratio, classes);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - begin;
        std::cout << "post process time: " << elapsed.count() << " ms" << std::endl;

        cv::imshow("TEST", resultImg);
        cv::imshow("TEST1", img_bgr);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}
