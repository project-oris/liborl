//
// $LICENSE
//
#include "json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "oris_orl_config.h"
#include "oris_orl_log.h"

using json = nlohmann::json;
using namespace oris::orl;

namespace oris
{
    namespace orl
    {

        options null_option;

        class JsonConfigManager
        {
        public:
            bool loadFile(const std::string &filename);
            bool saveFile(const std::string &filename) const;

            void setValue(const std::string &path, const nlohmann::json &value);
            nlohmann::json getValue(const std::string &path) const;
            bool hasPath(const std::string &path) const;
            std::string toString();

        private:
            nlohmann::json root;

            std::vector<std::string> splitPath(const std::string &path) const;
            nlohmann::json *getOrCreateNode(const std::vector<std::string> &keys);
            const nlohmann::json *getNode(const std::vector<std::string> &keys) const;
        };

        typedef struct _oris_orl_options_impl
        {
            JsonConfigManager options;
        } _oris_orl_options_impl;

        void __parse_config(config &_config)
        {
            _config.m_model = _config.m_options.get_str_option(AI_CONFIG_MODEL);
            _config.m_task = _config.m_options.get_str_option(AI_CONFIG_TASK);
            _config.m_device = _config.m_options.get_str_option(AI_CONFIG_DEVICE);
            _config.m_engine = _config.m_options.get_str_option(AI_CONFIG_ENGINE);
            _config.m_model_path = _config.m_options.get_str_option(AI_CONFIG_MODEL_PATH);
            _config.m_model_file_type = _config.m_options.get_str_option(AI_CONFIG_MODEL_FILE_TYPE);
            _config.m_driver_path = _config.m_options.get_str_option(AI_CNOFIG_DRIVER_PATH);

            // if (_config.m_engine.compare("orisai") == 0)
            // {
            //     _config.m_detect_height = _config.m_options.get_int_option(AI_CONFIG_DETECT_HEIGHT);
            //     _config.m_detect_width = _config.m_options.get_int_option(AI_CONFIG_DETECT_WIDTH);
            //     _config.m_normalization_value = _config.m_options.get_float_option(AI_CONFIG_NORMALIZATION_VALUE);
            // }

            // _config.m_yolo_nms = _config.m_options.get_bool_option(AI_CONFIG_YOLO_NMS);
        }

        /*
        shape_str : (1,2,224,224)
        */
        std::vector<int> parse_shape(const std::string &shape_str)
        {
            std::vector<int> shape;
            std::string cleanedString = shape_str;

            // 괄호 제거
            if (!cleanedString.empty() && cleanedString.front() == '(')
            {
                cleanedString.erase(0, 1);
            }
            if (!cleanedString.empty() && cleanedString.back() == ')')
            {
                cleanedString.pop_back();
            }

            // 쉼표를 기준으로 숫자 파싱
            std::stringstream ss(cleanedString);
            std::string segment;

            while (std::getline(ss, segment, ','))
            {
                try
                {
                    shape.push_back(std::stoi(segment));
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << "Invalid argument: " << segment << " - " << e.what() << std::endl;
                }
                catch (const std::out_of_range &e)
                {
                    std::cerr << "Out of range: " << segment << " - " << e.what() << std::endl;
                }
            }

            return shape;
        }

        bool load_config(config &_config, const std::string &_filepath)
        {
            try
            {
                _config.m_options.load_file(_filepath);

                // JsonConfigManager doc;

                // doc.loadFile(_filepath);

                __parse_config(_config);
            }
            catch (std::runtime_error &e)
            {
                log_error("load config error {}..{}", _filepath, e.what());
                return false;
            }
            return true;
        }

        options::options()
        {
            this->m_impl = nullptr;
        }

        options::options(const std::string &file_path)
        {
            this->m_impl = new _oris_orl_options_impl();
            this->m_impl->options.loadFile(file_path);
        }

        void options::load_file(const std::string &file_path)
        {
            if (this->m_impl == nullptr)
            {
                this->m_impl = new _oris_orl_options_impl();
            }
            this->m_impl->options.loadFile(file_path);
        }

        options::~options()
        {
            if (this->m_impl != nullptr)
            {
                delete this->m_impl;
            }
        }

        std::string options::get_str_option(const std::string &path, const std::string &default_value)
        {
            if (this->m_impl == nullptr)
                return std::string("");

            auto value = this->m_impl->options.getValue(path);
            if (value == json::value_t::null)
            {
                return default_value;
            }

            return value;
        }

        int options::get_int_option(const std::string &path, const int default_value)
        {
            if (this->m_impl == nullptr)
                return 0;

            auto value = this->m_impl->options.getValue(path);
            if (value == json::value_t::null)
            {
                return default_value;
            }

            return value;
        }

        float options::get_float_option(const std::string &path, const float default_value)
        {
            if (this->m_impl == nullptr)
                return 0.0;

            auto value = this->m_impl->options.getValue(path);
            if (value == json::value_t::null)
            {
                return default_value;
            }

            return value;
        }

        bool options::get_bool_option(const std::string &path, const bool default_value)
        {
            if (this->m_impl == nullptr)
                return 0.0;

            auto value = this->m_impl->options.getValue(path);
            if (value == json::value_t::null)
            {
                return default_value;
            }

            return value;
        }

        void options::set_str_option(const std::string &path, const std::string &value)
        {
            if (this->m_impl != nullptr)
            {
                this->m_impl->options.setValue(path, value);
            }
        }

        void options::set_int_option(const std::string &path, const int value)
        {
            if (this->m_impl != nullptr)
            {
                this->m_impl->options.setValue(path, value);
            }
        }

        void options::set_float_option(const std::string &path, const float value)
        {
            if (this->m_impl != nullptr)
            {
                this->m_impl->options.setValue(path, value);
            }
        }

        void options::set_bool_option(const std::string &path, const bool value)
        {
            if (this->m_impl != nullptr)
            {
                this->m_impl->options.setValue(path, value);
            }
        }

        bool options::has_option(const std::string &path)
        {
            if (this->m_impl != nullptr)
            {
                return this->m_impl->options.hasPath(path);
            }
            return false;
        }

        /////////////////////////////////////////////////////////
        //////   JsonConfigManager
        /////////////////////////////////////////////////////////

        std::vector<std::string> JsonConfigManager::splitPath(const std::string &path) const
        {
            std::vector<std::string> result;
            std::stringstream ss(path);
            std::string item;
            while (std::getline(ss, item, '/'))
            {
                if (!item.empty())
                    result.push_back(item);
            }
            return result;
        }

        bool JsonConfigManager::loadFile(const std::string &filename)
        {
            try
            {
                std::ifstream in(filename);
                if (!in.is_open())
                    return false;
                in >> root;
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        bool JsonConfigManager::saveFile(const std::string &filename) const
        {
            try
            {
                std::ofstream out(filename);
                if (!out.is_open())
                    return false;
                out << root.dump(4); // Pretty print with indent = 4
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        std::string JsonConfigManager::toString()
        {
            return root.dump(4);
        }

        json *JsonConfigManager::getOrCreateNode(const std::vector<std::string> &keys)
        {
            json *current = &root;
            for (const auto &key : keys)
            {
                current = &((*current)[key]);
            }
            return current;
        }

        const json *JsonConfigManager::getNode(const std::vector<std::string> &keys) const
        {
            const json *current = &root;
            for (const auto &key : keys)
            {
                if (current->contains(key))
                {
                    current = &((*current)[key]);
                }
                else
                {
                    return nullptr;
                }
            }
            return current;
        }

        void JsonConfigManager::setValue(const std::string &path, const json &value)
        {
            auto keys = splitPath(path);
            if (!keys.empty())
            {
                json *node = getOrCreateNode(keys);
                *node = value;
            }
        }

        json JsonConfigManager::getValue(const std::string &path) const
        {
            auto keys = splitPath(path);
            const json *node = getNode(keys);
            if (node)
            {
                return *node;
            }
            return json(); // return null json
        }

        bool JsonConfigManager::hasPath(const std::string &path) const
        {
            auto keys = splitPath(path);
            return getNode(keys) != nullptr;
        }

        /*


        int main() {
            std::string shapeStr1 = "(1,2,224,224)";
            std::vector<int> shape1 = parseShapeString(shapeStr1);

            std::cout << "Shape from \"" << shapeStr1 << "\": ";
            for (int dim : shape1) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            std::string shapeStr2 = "(10,5)";
            std::vector<int> shape2 = parseShapeString(shapeStr2);

            std::cout << "Shape from \"" << shapeStr2 << "\": ";
            for (int dim : shape2) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            std::string shapeStr3 = "(1)";
            std::vector<int> shape3 = parseShapeString(shapeStr3);

            std::cout << "Shape from \"" << shapeStr3 << "\": ";
            for (int dim : shape3) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            std::string shapeStr4 = "()"; // 빈 괄호 테스트
            std::vector<int> shape4 = parseShapeString(shapeStr4);

            std::cout << "Shape from \"" << shapeStr4 << "\": ";
            if (shape4.empty()) {
                std::cout << "empty";
            }
            for (int dim : shape4) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            std::string shapeStr5 = "(1,2,abc)"; // 잘못된 입력 테스트
            std::vector<int> shape5 = parseShapeString(shapeStr5);

            std::cout << "Shape from \"" << shapeStr5 << "\": ";
            for (int dim : shape5) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;

            return 0;
        }

        int main() {
            oris_orl_config config;

            if (!load_config(config, "oris_ai_config.json"))
            {
                std::cerr<<"ERR" <<std::endl;
                return -1;
            }

            std::cout << config.m_detect_height << std::endl;
            std::cout << config.m_detect_width << std::endl;
            std::cout << config.m_normalization_value << std::endl;


            // if (!load_config(config, "trt_config.json"))
            // {
            //     std::cerr<<"ERR" <<std::endl;
            //     return -1;
            // }

            return 0;
        }

        */
        /*
        #include <iostream>

        int main() {
            JsonConfigManager config;

            config.setValue("robot/name", "TurtleBot3");
            config.setValue("robot/sensors/camera", "Realsense D435");
            config.setValue("robot/sensors/lidar", "RPLidar A1");
            config.setValue("robot/max_speed", 1.2);

            config.saveFile("config.json");

            config.loadFile("config.json");
            std::cout << "Robot name: " << config.getValue("robot/name") << std::endl;
            std::cout << "Camera: " << config.getValue("robot/sensors/camera") << std::endl;

            if (config.hasPath("robot/max_speed")) {
                std::cout << "Max speed: " << config.getValue("robot/max_speed") << std::endl;
            }

            config.saveFile("config2.json");

            return 0;
        }
            */

    } // namespace orl

} //  namespace oris