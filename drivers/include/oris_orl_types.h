//
// $LICENSE
//

#ifndef _ORIS_ORL_TYPES_H_
#define _ORIS_ORL_TYPES_H_
#include <stdint.h>
#include <vector>
#include <string>
#include <cstddef>
#include <memory>

////////////////////////////////////////////////////

namespace oris
{
  namespace orl
  { // ondevice ros library

    enum class DataType
    {
      UINT8,
      INT16,
      INT32,
      INT64,
      FLOAT32,
      FLOAT64,
    };

    class _ivector_t
    {
    public:
      virtual ~_ivector_t() = default;
      virtual size_t size() const = 0;
    };

    template <typename T>
    class vector_t : public _ivector_t
    {
    private:
      std::vector<T> m_data;

    public:
      vector_t() = default;
      vector_t(size_t l) : m_data(l) {};
      vector_t(const T *f, const T *l) : m_data(f, l) {};
      ~vector_t() override {};
      size_t size() const override { return m_data.size(); }
      void reserve(size_t size) { m_data.reserve(size); }
      std::vector<T> &get() { return m_data; }
      const std::vector<T> &get() const { return m_data; }
    };

    struct Tensor
    {
      std::shared_ptr<_ivector_t> m_data;
      std::vector<size_t> shape;           // tensor shape
      DataType data_type{DataType::UINT8}; // Basic element types of tensor data
      std::string format{""};              // Data format information (e.g. such as CHW, HWC, NCHW, NHWC for image, etc.)
      std::vector<size_t> strides;         // Memory access interval for each dimension (in bytes)
    }; // struct TensorData

    ///////////////////////////////////////////////////
    size_t getTensorSizeInBytes(Tensor &_data);
    size_t get_volume_size(const std::vector<size_t> &dims);
    Tensor makeTensor(DataType type, std::vector<size_t> &dims);

  } // namespace orl
} // namespace

#endif // _ORIS_ORL_TYPES_H_