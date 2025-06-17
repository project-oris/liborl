/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute (ETRI)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *******************************************************************************/
#pragma once

#include <opencv2/opencv.hpp>
#include "oris_ai/model/model.h"

//--------------------------------------------
// Padding structure for image processing
//--------------------------------------------
struct Padding {
  int vertical = 0;    // top and bottom padding (same value)
  int horizontal = 0;  // left and right padding (same value)
};

//--------------------------------------------
// Thread control variables
//--------------------------------------------
extern bool run_thread;
extern bool terminated;

//--------------------------------------------
// Global variables for frame processing
//--------------------------------------------
extern cv::Mat img;
extern cv::Mat converted_img, input_image;
extern Padding padding;
extern float gain;

//--------------------------------------------
// ORIS AI runtime environment
//--------------------------------------------
extern std::unique_ptr<oris_ai::Model> model;
extern oris_ai::Device device;

//--------------------------------------------
// Default file paths
//--------------------------------------------
const std::string model_path = "../../test_model/yolov8n_oris_bn_merged_custom_5class.pb";
const std::string input_video_path = "input1.mp4";

//--------------------------------------------
// Global variable to store detection results
//--------------------------------------------
extern std::vector<oris_ai::Detection> show_detection_result;

//--------------------------------------------
// Colors for visualization
//--------------------------------------------
extern std::vector<cv::Scalar> colors;
cv::Scalar getColor(int class_id);

//--------------------------------------------
// Function declarations
//--------------------------------------------
void letterbox(const cv::Mat& image, cv::Mat& image_padded, Padding& padding,
  bool auto_pad = true, bool scale_fill = false,
  const cv::Size& target_size = cv::Size(640, 640),
  bool scale_up = true, bool center = true, int stride = 32,
  const cv::Scalar& color = cv::Scalar(114, 114, 114));

void detection_thread_func();
void warmup(size_t org_frame_height, size_t org_frame_width, size_t detect_height, size_t detect_width);

//--------------------------------------------
// Configurations for detection
//--------------------------------------------
constexpr size_t detect_height = 384;
constexpr size_t detect_width = 640;