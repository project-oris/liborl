//
// $LICENSE
//

#ifndef _ORIS_ORL_CONFIG_H_
#define _ORIS_ORL_CONFIG_H_
#include <string>
#include <vector>
// #ifdef __cplusplus
// extern "C"
// {
// #endif

////////////////////////////////////////////////////

namespace oris
{
    namespace orl
    {

        /////////////////////////////////////////
        // HARD CORDED CONFIG VALUE
        /////////////////////////////////////////

#define AI_CONFIG_MODEL "model"   // yolov8n, etc.
#define AI_CONFIG_TASK "task"     // detection, classification
#define AI_CONFIG_DEVICE "device" // cpu, gpu
#define AI_CONFIG_ENGINE "engine" // trt, orisai, mobilint, etc.

#define AI_CONFIG_MODEL_PATH "model_path"           //
#define AI_CONFIG_MODEL_FILE_TYPE "model_file_type" // trt, onnx, orisai, mobilint, etc.
#define AI_CNOFIG_DRIVER_PATH "driver_path"

// only for oris_ai
#define AI_CONFIG_ORISAI_NORMALIZATION_VALUE "orisai/normalization_value" //

#define AI_CONFIG_TENSOR_NORMALIZATION "tensor/normalization"           // if true, normalization is applied
#define AI_CONFIG_TENSOR_INPUT_SHAPE "tensor/input_shape"               // (height, width)
#define AI_CONFIG_TENSOR_NORMALIZATION_BASE "tensor/normalization_base" // normalization base . if rgb image, it has 255.0

// parameter for
#define AI_CONFIG_YOLO_NMS "yolo/nms"     // model has nms
#define AI_CONFIG_YOLO_TOCHW "yolo/tochw" // convert input(HWC) --> CHW (channel, height, width)

        ////////////////////////////

        // task type(yolo)
        // detection  : yolo11
        // segmentation  : yolo11-seg
        // pose  : yolo11-pose
        // oriented detection : yolo11-obb
        // classification  : yolo11-cls

#define AI_TASK_DETECTION "detection"
#define AI_TASK_CLASSIFICAATION "classification"
#define AI_TASK_POSE_ESTIMATION "pose"
#define AI_TASK_OBB_DETECTION "obb" // oriented bounding boxes
#define AI_TASK_SEGMENTATION "segmentation"

        // model: yolo, resnet, etc...

        typedef struct _oris_orl_options_impl *_options_impl_ptr;

        class options
        {
        public:
            options();
            options(const std::string &file_path);
            virtual ~options();
            std::string get_str_option(const std::string &path, const std::string &default_value = "");
            int get_int_option(const std::string &path, const int default_value = 0);
            float get_float_option(const std::string &path, const float default_value = 0.0);
            bool get_bool_option(const std::string &path, const bool default_value = false);
            void load_file(const std::string &file_path);
            void set_str_option(const std::string &path, const std::string &value);
            void set_int_option(const std::string &path, const int value);
            void set_float_option(const std::string &path, const float value);
            void set_bool_option(const std::string &path, const bool value);
            bool has_option(const std::string &path);

        private:
            _options_impl_ptr m_impl{nullptr};
        };

        extern options null_option;

        std::vector<int> parse_shape(const std::string &shape_str);

        typedef struct config
        {
            std::string m_model;
            std::string m_task;
            std::string m_device;
            std::string m_engine;
            std::string m_model_path;
            std::string m_model_file_type;
            std::string m_driver_path;

            // int m_detect_height;
            // int m_detect_width;
            // float m_normalization_value{255.0};
            // bool m_yolo_nms{false};
            options m_options;
        } config;

        bool load_config(config &_config, const std::string &_filepath);

    } //      namespace orl

} // namespace oris

// #ifdef __cplusplus
// } // extern "C"
// #endif

#endif // _ORIS_ORL_CONFIG_H_