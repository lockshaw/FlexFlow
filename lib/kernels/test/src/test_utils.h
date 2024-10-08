#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include <doctest/doctest.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace FlexFlow;

template <typename DT>
void transfer_memory(GenericTensorAccessorW dst_accessor,
                     const DT *src,
                     DeviceType src_device_type) {
  size_t bytes = dst_accessor.shape.get_volume() * sizeof(DT);

  DeviceType dst_device_type = dst_accessor.device_type;

  if (device_on_cpu(src_device_type) && device_on_cpu(dst_device_type)) {
    memcpy(dst_accessor.ptr, src, bytes);
  } else if (device_on_cpu(src_device_type) && device_on_gpu(dst_device_type)) {
    checkCUDA(cudaMemcpy(dst_accessor.ptr, src, bytes, cudaMemcpyHostToDevice));
  } else if (device_on_gpu(src_device_type) && device_on_cpu(dst_device_type)) {
    checkCUDA(cudaMemcpy(dst_accessor.ptr, src, bytes, cudaMemcpyDeviceToHost));
  } else {
    checkCUDA(
        cudaMemcpy(dst_accessor.ptr, src, bytes, cudaMemcpyDeviceToDevice));
  }
}

template <DataType DT>
GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator) {
  assert(shape.data_type == DataType::FLOAT ||
         shape.data_type == DataType::DOUBLE);

  using T = real_type_t<DT>;

  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);

  std::vector<T> host_data(accessor.shape.num_elements());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dist(-1.0, 1.0);

  for (auto &val : host_data) {
    val = dist(gen);
  }

  transfer_memory(accessor, host_data.data(), DeviceType::CPU);

  return accessor;
}

template <DataType DT>
GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator) {
  using T = real_type_t<DT>;
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w<DT>(shape, allocator);

  return read_only_accessor_from_write_accessor(accessor);
}

template <typename T>
GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                T val) {
  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);

  size_t volume = accessor.shape.get_volume();
  std::vector<T> host_data(volume, val);

  transfer_memory(accessor, host_data.data(), DeviceType::CPU);

  return accessor;
}

template <DataType DT>
std::vector<real_type_t<DT>>
    load_accessor_data(GenericTensorAccessorR accessor) {
  using T = real_type_t<DT>;

  int volume = accessor.shape.get_volume();
  std::vector<T> local_data(volume);
  T const *src_ptr = accessor.get<DT>();

  if (device_on_cpu(accessor.device_type)) {
    memcpy(local_data.data(), src_ptr, volume * sizeof(T));
  } else {
    checkCUDA(cudaMemcpy(local_data.data(),
                         src_ptr,
                         volume * sizeof(T),
                         cudaMemcpyDeviceToHost));
  }

  return local_data;
}

template <DataType DT>
std::vector<real_type_t<DT>>
    load_accessor_data(GenericTensorAccessorW accessor) {
  using T = real_type_t<DT>;

  int volume = accessor.shape.get_volume();
  std::vector<T> local_data(volume);
  T const *src_ptr = accessor.get<DT>();

  if (device_on_cpu(accessor.device_type)) {
    memcpy(local_data.data(), src_ptr, volume * sizeof(T));
  } else {
    checkCUDA(cudaMemcpy(local_data.data(),
                         src_ptr,
                         volume * sizeof(T),
                         cudaMemcpyDeviceToHost));
  }

  return local_data;
}

template <typename T>
bool contains_non_zero(std::vector<T> &data) {
  return !all_of(
      data.begin(), data.end(), [](T const &val) { return val == 0; });
}

template <typename T, typename Func>
std::vector<T> repeat(std::size_t n, Func &&func) {
  std::vector<T> result;
  // result.reserve(n); // Sometimes we don't have default constructor for T
  for (std::size_t i = 0; i < n; ++i) {
    result.push_back(func());
  }
  return result;
}

// Specialize doctest's StringMaker for std::vector<float>
template <>
struct doctest::StringMaker<std::vector<float>> {
  static doctest::String convert(std::vector<float> const &vec) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i != vec.size() - 1) {
        oss << ", ";
      }
    }
    return doctest::String(("[" + oss.str() + "]").c_str());
  }
};

#endif
