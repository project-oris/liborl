#include <iostream>
#include <dlfcn.h> // POSIX (Linux, macOS)
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <mutex>
#include "oris_orl_driver.h"
#include "oris_orl_types.h"
#include "oris_orl_config.h"
#include "oris_orl_cv_support.h"
#include "oris_orl_yolo_support.h"

oris::orl::config g_rt_config;
oris::orl::driver_t driver_internal = {0, 0, 0, 0, 0};
const std::string input_video_path = "input1.mp4";
cv::Mat img;
cv::Mat curr_img;
// std::vector<std::string> classes{"person", "car", "ev charging station", "cctv", "gas station"};
std::vector<std::string> classes{"person", "car", "ev charging station", "cctv", "gas station", "bus", "train", "truck", "boat", "traffic light",
                                 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                 "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                 "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                 "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                                 "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                 "scissors", "teddy bear", "hair drier", "toothbrush"};

// std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//                                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//                                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//                                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//                                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//                                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//                                  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
//                                  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
//                                  "scissors", "teddy bear", "hair drier", "toothbrush"};

bool run_thread = false;
bool terminated = false;
cv::Mat output_frame;
std::vector<oris::yolo::detection_result_t> show_detect_result;

std::map<std::string, std::vector<size_t>> InputShapes;
std::map<std::string, std::vector<size_t>> OutputShapes;

bool useImageTensor = false;
bool useToCHW = false;
int target_height = 384;
int target_width = 640;
std::mutex mtx;
std::mutex result_mtx;

float yolo_preprocess(const cv::Mat &input, cv::Mat &converted, size_t height, size_t width, const cv::Scalar &bgcolor);

void detection_thread_func()
{
  if (driver_internal.engine_handle == NULL)
  {
    std::cerr << "Driver Cannt be null" << std::endl;
    return;
  }

  while (!terminated)
  {

    std::vector<oris::yolo::detection_result_t> detect_result;
    oris::orl::Tensor img_tensor;
    oris::orl::options param;
    std::vector<oris::orl::Tensor> input_tensors;
    std::vector<oris::orl::Tensor> result;
    cv::Mat converted_img;

    if (img.empty())
      continue;
    try
    {

      // preprocess
      float ratio;
      {
        std::unique_lock<std::mutex> lock(mtx);
        curr_img = img;        
        output_frame = curr_img.clone();
        ratio = oris::yolo::preprocess(curr_img, converted_img, target_height, target_width, cv::Scalar(0, 0, 0));
      }
      

      std::cerr << " org size is " << curr_img.size().width << "," << curr_img.size().height << std::endl;
      std::cerr << " img size is " << converted_img.size().width << "," << converted_img.size().height << std::endl;
      std::cerr << "ratio is " << ratio << std::endl;

      // cvMat To Tensor
      if (useImageTensor)
      { // oris-ai
        cvMatToTensor(converted_img, img_tensor, useToCHW);
        // cvMatToFloatTensor(converted_img, img_tensor, false, 1.0);
      }
      else
      {
        // cvMatToFloatTensor(converted_img, img_tensor, true, 255.0);
        cvMatToTensor(converted_img, img_tensor, true);
      }

      input_tensors.push_back(img_tensor);

      oris::orl::run_infer(driver_internal, input_tensors, result);

      if (result.size() > 0)
      {
        std::vector<oris::yolo::detection_result_t> d_result;
        oris::yolo::tensor_to_detection_result(result[0], d_result);
        for (auto &item : d_result)
        {
          item.x1 = std::clamp((int)(item.x1 * ratio), 0, curr_img.cols);
          item.y1 = std::clamp((int)(item.y1 * ratio), 0, curr_img.rows);
          item.x2 = std::clamp((int)(item.x2 * ratio), 0, curr_img.cols);
          item.y2 = std::clamp((int)(item.y2 * ratio), 0, curr_img.rows);
        }
        show_detect_result = d_result;
        // post p
      }
    }
    catch (std::bad_alloc &ex)
    {
      std::cerr << "ERR" << ex.what() << std::endl;
      exit(1);
    }
  }
}

std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 255, 0),    // Green
    cv::Scalar(189, 114, 0),  // Blue
    cv::Scalar(32, 177, 237), // Yellow
    cv::Scalar(142, 47, 126), // Purple
    cv::Scalar(0, 0, 255)     // Red
};

cv::Scalar getColor(int class_id)
{
  int index = class_id % colors.size();
  return colors[index];
}

int main()
{
  std::string config_file_name{"./config.json"};
  if (!oris::orl::load_driver(driver_internal, config_file_name))
  {
    std::cerr << "cannot load driver " << config_file_name << std::endl;
    return -1;
  }

  get_input_shape(driver_internal, InputShapes);
  get_output_shape(driver_internal, OutputShapes);

  for (auto &a : InputShapes)
  {
    std::cerr << a.first << "..(";
    for (auto &b : a.second)
    {
      std::cerr << b << ",";
    }
    std::cerr << std::endl;
  }

  for (auto &a : OutputShapes)
  {
    std::cerr << a.first << "..(";
    for (auto &b : a.second)
    {
      std::cerr << b << ",";
    }
    std::cerr << ")" << std::endl;
  }

  std::map<std::string, std::vector<size_t>> InputShapes;
  std::map<std::string, std::vector<size_t>> OutputShapes;

  if (!driver_internal.m_config.m_engine.compare("orisai"))
  {
    useImageTensor = true;
    useToCHW = false;
    target_height = 384;
    target_width = 640;
  }
  else
  {
    useImageTensor = false;
    useToCHW = true;
    target_height = 640;
    target_width = 640;
  }

  oris::orl::options run_options;
  // Open video file
  cv::VideoCapture cap(input_video_path);
  if (!cap.isOpened())
  {
    std::cerr << "Error: Unable to open input video file: " << input_video_path << std::endl;
    return -1;
  }

  std::thread inference_thread(detection_thread_func);
  inference_thread.detach();
  run_thread = true;

  while (cap.read(img))
  {
    if (img.empty())
    {
      /* end video */
      terminated = true;
      run_thread = false;
      cv::Mat dummy_image(384, 640, CV_8UC3, cv::Scalar(0, 0, 0));
      img = dummy_image.clone();
      std::cout << "-----------------------------" << std::endl;
      std::cout << "end the video..." << std::endl;
      std::cout << "-----------------------------" << std::endl;
      sleep(1);
      break;
    }
    else
    {
      // if (mtx.try_lock())
      // {
      //   curr_img = img.clone();
      //   mtx.unlock();
      // }
    }

    // if (!run_thread && !terminated)
    // {
    //   std::thread inference_thread(detection_thread_func);
    //   inference_thread.detach();
    //   run_thread = true;
    // }

    {
      // std::unique_lock<std::mutex> lock(result_mtx);
       std::unique_lock<std::mutex> lock(mtx);

      if (!output_frame.empty())
      {

        for (const auto &detection : show_detect_result)
        {
          int class_id = detection.class_id;
          int x1 = detection.x1;
          int y1 = detection.y1;
          int x2 = detection.x2;
          int y2 = detection.y2;
          cv::Scalar color = getColor(class_id);
          cv::rectangle(output_frame, cv::Point(x1, y1),
                        cv::Point(x2, y2), color, 2);
          std::string label = classes[class_id] + ": " + std::to_string(detection.confidence * 100).substr(0, 5) + "%";
          int baseline;
          cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
          y1 = std::max(y1, textSize.height);
          cv::rectangle(output_frame, cv::Point(x1, y1 - textSize.height - baseline),
                        cv::Point(x1 + textSize.width, y1), color, cv::FILLED);
          cv::putText(output_frame, label, cv::Point(x1, y1 - baseline),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
      }
    }

    if (!output_frame.empty())
    {
      cv::imshow("ORIS AI Demo", output_frame);
      char key = cv::waitKey(30);

      if (key == 27)
      {
        // stop capturing by pressing ESC
        std::cout << "close the video..." << std::endl;
        terminated = true;
        run_thread = false;
        sleep(1);
        break;
      }
    }

  } // end of while

  terminated = true; // Ensure thread termination
  run_thread = false;
  sleep(1); // Give some time for the thread to finish

  cv::destroyAllWindows();
  cap.release();

  oris::orl::unload_driver(driver_internal);
}