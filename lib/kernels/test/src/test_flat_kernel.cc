#include "kernels/flat_kernels_gpu.h"
#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/accessor_contains_non_zero_value.h"
#include "kernels/create_constant_filled_accessor_r.h"
#include "kernels/accessor.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    Allocator allocator = create_local_cuda_memory_allocator();

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(create_constant_filled_accessor_w(
            input_shape, allocator, make_float_data_type_value(2)));

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Flat::gpu_forward_kernel(managed_stream.raw_stream(),
                                        input_accessor,
                                        output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR output_grad_accessor = create_constant_filled_accessor_r(
          output_shape, allocator, make_float_data_type_value(0));
      GenericTensorAccessorW input_grad_accessor = create_constant_filled_accessor_w(
          input_shape, allocator, make_float_data_type_value(1));

      Kernels::Flat::gpu_backward_kernel(managed_stream.raw_stream(),
                                         input_accessor,
                                         output_grad_accessor.get_float_ptr(),
                                         input_grad_accessor.get_float_ptr());

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
