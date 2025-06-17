//
// $LICENSE
//
#include "oris_orl_types.h"

using namespace oris::orl;

namespace oris
{
  namespace orl
  {

    size_t get_volume_size(const std::vector<size_t> &dims)
    {
      size_t sums = 1;
      for (auto &a : dims)
      {
        sums *= a;
      }
      return sums;
    }

    Tensor makeTensor(DataType type, std::vector<size_t> &dims)
    {
      Tensor result;

      result.shape = dims;
      result.data_type = type;
      size_t totalsize = get_volume_size(dims);
      switch (type)
      {
      case DataType::UINT8:
        result.m_data = std::make_shared<vector_t<uint8_t>>(totalsize);
        break;
      case DataType::INT16:
        result.m_data = std::make_shared<vector_t<int16_t>>(totalsize);
        break;
      case DataType::INT32:
        result.m_data = std::make_shared<vector_t<int32_t>>(totalsize);
        break;
      case DataType::INT64:
        result.m_data = std::make_shared<vector_t<int64_t>>(totalsize);
        break;
      case DataType::FLOAT32:
        result.m_data = std::make_shared<vector_t<float>>(totalsize);
        break;
      case DataType::FLOAT64:
        result.m_data = std::make_shared<vector_t<double>>(totalsize);
        break;
      }

      return result;
    }

    size_t getTensorSizeInBytes(Tensor &_data)
    {
      if (_data.shape.empty())
        return 0;
      size_t size = 1;
      for (int64_t dim : _data.shape)
      {
        size *= dim;
      }
      size_t element_size = 0;
      switch (_data.data_type)
      {
      case DataType::UINT8:
        element_size = sizeof(uint8_t);
        break;
      case DataType::INT16:
        element_size = sizeof(int16_t);
        break;
      case DataType::INT32:
        element_size = sizeof(int32_t);
        break;
      case DataType::INT64:
        element_size = sizeof(int64_t);
        break;
      case DataType::FLOAT32:
        element_size = sizeof(float);
        break;
      case DataType::FLOAT64:
        element_size = sizeof(double);
        break;
        // 다른 데이터 타입에 대한 처리 추가
      }
      return size * element_size;
    }

  } // namespace orl
} // namespace oris