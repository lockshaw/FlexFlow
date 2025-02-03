#include "kernels/attention_kernels.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "utils/containers/reversed.h"
#include <benchmark/benchmark.h>
#include "kernels/local_cuda_allocator.h"
#include "benchmark/utils/suite.h"

using namespace ::FlexFlow;

static void benchmark_cuda_attention_kernel(benchmark::State &state) {
  nonnegative_int num_samples = 10_n;
  nonnegative_int num_heads = 4_n;
  nonnegative_int qSize = 64_n;
  nonnegative_int kSize = 64_n;
  nonnegative_int vSize = 64_n;
  nonnegative_int qProjSize = 64_n;
  nonnegative_int kProjSize = 64_n;
  nonnegative_int vProjSize = 64_n;
  nonnegative_int oProjSize = 64_n;
  nonnegative_int qoSeqLength = 20_n;
  nonnegative_int kvSeqLength = 20_n;

  ManagedFFStream managed_stream{};
  ManagedPerDeviceFFHandle managed_handle{};

  Allocator allocator = create_local_cuda_memory_allocator();

  MHAPerDeviceState per_device_state = Kernels::MultiHeadAttention::init_kernel(
      managed_handle.raw_handle(),
      allocator,
      /*num_samples=*/num_samples.unwrap_nonnegative(),
      /*num_heads=*/num_heads.unwrap_nonnegative(),
      /*qSize=*/qSize.unwrap_nonnegative(),
      /*kSize=*/kSize.unwrap_nonnegative(),
      /*vSize=*/vSize.unwrap_nonnegative(),
      /*qProjSize=*/qProjSize.unwrap_nonnegative(),
      /*kProjSize=*/kProjSize.unwrap_nonnegative(),
      /*vProjSize=*/vProjSize.unwrap_nonnegative(),
      /*oProjSize=*/oProjSize.unwrap_nonnegative(),
      /*qoSeqLength=*/qoSeqLength.unwrap_nonnegative(),
      /*kvSeqLength=*/kvSeqLength.unwrap_nonnegative(),
      /*add_bias_kv=*/false);

  auto make_float_tensor_shape_from_legion_dims = [](std::vector<nonnegative_int> const &dims) {
    return TensorShape{TensorDims{FFOrdered<nonnegative_int>{reversed(dims)}}, DataType::FLOAT};

  };

  TensorShape query_shape = make_float_tensor_shape_from_legion_dims(
      {qoSeqLength, num_samples, qSize});
  TensorShape key_shape = make_float_tensor_shape_from_legion_dims(
      {kvSeqLength, num_samples, kSize});
  TensorShape value_shape = make_float_tensor_shape_from_legion_dims(
      {kvSeqLength, num_samples, vSize});
  TensorShape output_shape = make_float_tensor_shape_from_legion_dims(
      {qoSeqLength, num_samples, oProjSize});
  TensorShape weight_shape = make_float_tensor_shape_from_legion_dims(
      {nonnegative_int{per_device_state.weightSize}});

  GenericTensorAccessorW query_accessor =
      allocator.allocate_tensor(query_shape);
  GenericTensorAccessorW key_accessor =
      allocator.allocate_tensor(key_shape);
  GenericTensorAccessorW value_accessor =
      allocator.allocate_tensor(value_shape);
  GenericTensorAccessorW weight_accessor =
      allocator.allocate_tensor(weight_shape);

  GenericTensorAccessorW output_accessor =
      allocator.allocate_tensor(output_shape);

  for (auto _ : state) {
    
  }
}

FF_CUDA_BENCHMARK(benchmark_cuda_attention_kernel);
