#include <map>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // std::vector, std::map 변환
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "oris_orl_types.h"
#include "oris_orl_config.h"
#include "oris_orl_driver.h"
#include "oris_orl_cv_support.h"
// #include "oris_orl_yolo_support.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace oris::orl;

cv::Mat numpy_to_cvmat(py::array array)
{
    py::buffer_info info = array.request();
    int type;
    if (info.format == pybind11::format_descriptor<uint8_t>::format())
    {
        type = CV_8U;
    }
    else if (info.format == pybind11::format_descriptor<float>::format())
    {
        type = CV_32F;
    }
    else
    {
        throw std::runtime_error("Unsupported data type: " + info.format);
    }

    int ndims = info.ndim;
    if (ndims < 2 || ndims > 3)
    {
        throw std::runtime_error("Expected 2D or 3D array");
    }

    int height = static_cast<int>(info.shape[0]);
    int width = static_cast<int>(info.shape[1]);
    int channels = (ndims == 3) ? static_cast<int>(info.shape[2]) : 1;

    // OpenCV type 설정 (예: CV_8UC3, CV_32FC1 등)
    int cv_type = CV_MAKETYPE(type, channels);

    // cv::Mat 생성 (데이터는 복사하지 않고 공유)
    return cv::Mat(height, width, cv_type, info.ptr);
}

Tensor numpy_to_tensor(py::array array)
{
    Tensor t;
    t.shape = std::vector<size_t>(array.shape(), array.shape() + array.ndim());
    t.strides = std::vector<size_t>(array.strides(), array.strides() + array.ndim());

    if (array.dtype().kind() == 'f' && array.itemsize() == 4)
    {
        t.data_type = DataType::FLOAT32;
        t.m_data = std::make_shared<vector_t<float>>(
            (float *)array.data(), (float *)array.data() + array.size());

#ifdef MODULE_DEBUG
        std::cerr << " Tensor os float";

        auto wrapper = dynamic_cast<vector_t<float> *>(t.m_data.get());
        int z = 0;
        for (auto adata : wrapper->get())
        {
            if (z > 20)
                break;
            z++;
            std::cerr << adata << ",";
        }

        // cv::Mat 생성 (데이터는 복사하지 않고 공유)
        cv::Mat mdata = numpy_to_cvmat(array);

        std::cerr << "Show Im Show...!!" << std::endl;
        // cv::imshow("demo", mdata);

        // cv::waitKey(0);
#endif //  MODULE_DEBUG
    }
    else if (array.dtype().kind() == 'u' && array.itemsize() == 1)
    {

        t.data_type = DataType::UINT8;
        t.m_data = std::make_shared<vector_t<uint8_t>>(
            (uint8_t *)array.data(), (uint8_t *)array.data() + array.size());
#ifdef MODULE_DEBUG
        std::cerr << " Tensor os UINT8";

        // cv::Mat 생성 (데이터는 복사하지 않고 공유)
        cv::Mat mdata = numpy_to_cvmat(array);

        std::cerr << "Show Im Show...!!" << std::endl;
        // cv::imshow("demo", mdata);

        // cv::waitKey(0);
#endif // MODULE_DEBUG
    }
    else
    {
        throw std::runtime_error("Unsupported NumPy dtype");
    }
    return t;
}

py::array tensor_to_numpy(const Tensor &t)
{
    py::dtype dtype;

    std::vector<ssize_t> shape(t.shape.begin(), t.shape.end());
    std::vector<ssize_t> strides(t.strides.begin(), t.strides.end());

    void *data_ptr;

    if (t.data_type == DataType::FLOAT32)
    {
        dtype = py::dtype::of<float>();
        auto wrapper = dynamic_cast<vector_t<float> *>(t.m_data.get());

        data_ptr = (void *)(wrapper->get().data());
    }
    else if (t.data_type == DataType::UINT8)
    {
        dtype = py::dtype::of<uint8_t>();
        auto wrapper = dynamic_cast<vector_t<float> *>(t.m_data.get());
        data_ptr = (void *)(wrapper->get().data());
    }
    else
    {
        throw std::runtime_error("Unsupported Tensor data_type");
    }

    return py::array(dtype, shape, strides, data_ptr);
}

// Python wrapper 함수
py::list py_run_infer(driver_t &driver, const std::vector<py::array> &py_inputs)
{
    std::vector<Tensor> inputs;
    float ratio = 1.0;
    cv::Mat curr_img;
    cv::Mat converted_img;

    bool tochw = driver.m_config.m_options.get_bool_option(AI_CONFIG_YOLO_TOCHW);

    for (auto &arr : py_inputs)
    {
        cv::Mat img = numpy_to_cvmat(arr);
        curr_img = img.clone();

        // ratio = oris::yolo::preprocess(img, converted_img, 640, 640, cv::Scalar(0, 0, 0));
        converted_img = curr_img;
        oris::orl::Tensor img_tensor;

        // cvMatToFloatTensor(converted_img, img_tensor, true, 255.0);
        // cvMatToFloatTensor(converted_img, img_tensor, false, 1.0);

        

        cvMatToTensor(converted_img, img_tensor, tochw);

        // cvMatToTensor(converted_img, img_tensor,false);

        inputs.push_back(img_tensor);
    }

    // std::cerr << "RATIO" << ratio << std::endl;

    std::vector<Tensor> results;
    int ret = run_infer(driver, inputs, results, null_option);
    // if (ret != 0)
    // {
    //     throw std::runtime_error("Inference failed with code: " + std::to_string(ret));
    // }

#ifdef DEV_DEBUG

    std::cerr << " REULST VALUE IS " << results.size() << std::endl;

    if (results.size() > 0)
    {
        std::vector<oris::yolo::detection_result_t> d_result;
        oris::yolo::tensor_to_detection_result(results[0], d_result);
        for (auto &item : d_result)
        {
            item.x1 = std::clamp((int)(item.x1 * ratio), 0, curr_img.cols);
            item.y1 = std::clamp((int)(item.y1 * ratio), 0, curr_img.rows);
            item.x2 = std::clamp((int)(item.x2 * ratio), 0, curr_img.cols);
            item.y2 = std::clamp((int)(item.y2 * ratio), 0, curr_img.rows);
        }

        std::cerr << "Total result is " << d_result.size() << std::endl;

        /* EXPEXTED

        (130,219)-(308,541)-(0.912208)..[16]
        (129,138)-(567,420)-(0.882803)..[1]
        (466,74)-(692,171)-(0.53549)..[2]
        (466,74)-(692,171)-(0.503707)..[7]
        (0,0)-(0,0)-(0)..[0]
        (0,0)-(0,0)-(0)..[0]
        (0,0)-(0,0)-(0)..[0]

        */

        for (int i = 0, j = 0; i < d_result.size(); i++, j++)
        {
            if (j > 6)
                break;
            std::cerr << "("
                      << d_result[i].x1 << ","
                      << d_result[i].y1 << ")-("
                      << d_result[i].x2 << ","
                      << d_result[i].y2 << ")-("
                      << d_result[i].confidence << ")..["
                      << d_result[i].class_id << "]" << std::endl;
        }

        // post p
    }
#endif // DEV_DEBUG

    py::list py_results;
    for (auto &t : results)
    {
        py_results.append(tensor_to_numpy(t));
    }
    return py_results;
}

py::dict py_get_input_shape(driver_t &driver)
{
    std::map<std::string, std::vector<size_t>> shape_map;
    int ret = get_input_shape(driver, shape_map);

    // C++ map -> Python dict 변환
    py::dict py_result;
    for (const auto &kv : shape_map)
    {
        py_result[py::str(kv.first)] = py::cast(kv.second);
    }

    return py_result;
}

int py_get_int_option(driver_t &driver, const std::string &path)
{
    return driver.m_config.m_options.get_int_option(path);
}

float py_get_float_option(driver_t &driver, const std::string &path)
{
    return driver.m_config.m_options.get_float_option(path);
}

std::string py_get_string_option(driver_t &driver, const std::string &path)
{
    return driver.m_config.m_options.get_str_option(path);
}

bool py_get_bool_option(driver_t &driver, const std::string &path)
{
    return driver.m_config.m_options.get_bool_option(path);
}

py::dict py_get_output_shape(driver_t &driver)
{
    std::map<std::string, std::vector<size_t>> shape_map;
    int ret = get_output_shape(driver, shape_map);

    // C++ map -> Python dict 변환
    py::dict py_result;
    for (const auto &kv : shape_map)
    {
        py_result[py::str(kv.first)] = py::cast(kv.second);
    }

    return py_result;
}

// Python-friendly wrapper
py::object py_load_driver(const std::string &config_path)
{
    auto driver = std::make_unique<driver_t>();
    bool ok = load_driver(*driver, config_path);
    if (!ok)
    {
        throw std::runtime_error("Failed to load driver with config: " + config_path);
    }
    return py::cast(driver.release()); // unique_ptr → raw pointer → Python object
}

PYBIND11_MODULE(oris_orl_py, m)
{
    m.doc() = "orl driver binding module";
    py::class_<driver_t>(m, "Driver")
        .def(py::init<>()); // default constructor

    py::enum_<DataType>(m, "DataType")
        .value("UINT8", DataType::UINT8)
        .value("INT16", DataType::INT16)
        .value("INT32", DataType::INT32)
        .value("INT64", DataType::INT64)
        .value("FLOAT32", DataType::FLOAT32)
        .value("FLOAT64", DataType::FLOAT64)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("data_type", &Tensor::data_type)
        .def_readwrite("format", &Tensor::format)
        .def_readwrite("strides", &Tensor::strides);

    m.def("makeTensor", &makeTensor, py::arg("data_type"), py::arg("dims"),
          "Create a new Tensor with given data type and dimensions");

    m.def("load_driver", &py_load_driver,
          py::arg("config_path"),
          "Load driver from config file and return a Driver object");

    // m.def("load_driver", &load_driver, py::arg("driver"), py::arg("config_path"),
    //       "Load driver with config");

    m.def("get_input_shape", &py_get_input_shape,
          py::arg("driver"),
          "Return a dictionary of input tensor shapes");

    m.def("get_output_shape", &py_get_output_shape,
          py::arg("driver"),
          "Return a dictionary of output tensor shapes");

    m.def("run_infer", &py_run_infer, py::arg("driver"), py::arg("inputs"),
          "Run inference and return list of numpy arrays");

    m.def("get_int_option", &py_get_int_option, py::arg("driver"), py::arg("key"),
          "Return int option value");

    m.def("get_float_option", &py_get_float_option, py::arg("driver"), py::arg("key"),
          "Return float option value");

    m.def("get_string_option", &py_get_string_option, py::arg("driver"), py::arg("key"),
          "Return string option value");

    m.def("get_bool_option", &py_get_bool_option, py::arg("driver"), py::arg("key"),
          "Return bool option value");
}
