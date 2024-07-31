#include "doctest/doctest.h"
#include "kernels/replicate_kernels.h"
#include "kernels/replicate_kernels_cpu.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Call Replicate Forward and Backward Kernels") {
    std::size_t num_replicas = 10;

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({100}, DataType::FLOAT);
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims({100}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Replicate::forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Replicate::backward_kernel(managed_stream.raw_stream(),
                                          output_grad_accessor,
                                          input_grad_accessor,
                                          num_replicas);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }

  TEST_CASE("Check Replicate Forward and Backward Kernel against CPU Kernel") {
    std::size_t num_replicas = 2;

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({5}, DataType::FLOAT);
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims({5, num_replicas}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("forward_kernel") {
      // Run GPU Replicate Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r(input_shape, gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          create_zero_filled_accessor_w(output_shape, gpu_allocator);

      Kernels::Replicate::forward_kernel(
          managed_stream.raw_stream(), input_accessor_gpu, output_accessor_gpu);

      // Run CPU Replicate Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          create_zero_filled_accessor_w(output_shape, cpu_allocator);

      Kernels::Replicate::cpu_forward_kernel(input_accessor_cpu,
                                             output_accessor_cpu);

      CHECK(accessors_are_equal(output_accessor_gpu, output_accessor_cpu));
    }

    SUBCASE("backward_kernel") {
      // Run GPU Replicate Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r(output_shape, gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          create_zero_filled_accessor_w(input_shape, gpu_allocator);

      Kernels::Replicate::backward_kernel(managed_stream.raw_stream(),
                                          output_grad_accessor_gpu,
                                          input_grad_accessor_gpu,
                                          num_replicas);

      // Run CPU Replicate Backward Kernel
      GenericTensorAccessorR output_grad_accessor_cpu =
          copy_tensor_accessor_r(output_grad_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          create_zero_filled_accessor_w(input_shape, cpu_allocator);

      Kernels::Replicate::cpu_backward_kernel(
          output_grad_accessor_cpu, input_grad_accessor_cpu, num_replicas);

      CHECK(accessors_are_equal(input_grad_accessor_gpu,
                                input_grad_accessor_cpu));
    }
  }

  TEST_CASE("Check Replicate Forward Kernel against CPU Kernel") {
    std::size_t num_replicas = 10;

    // This should be like three shapes: pre_replication, replication shape, and
    // reduced shape, but things are weird cause doesn't seem to be replicating
    // anything (ie. input shape should be same as reduced shape)
    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({10, num_replicas}, DataType::FLOAT);
    TensorShape replicated_shape =
        make_tensor_shape_from_legion_dims({10, num_replicas}, DataType::FLOAT);
    TensorShape reduced_shape =
        make_tensor_shape_from_legion_dims({10}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("forward_kernel") {
      // Run GPU Replicate Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r<DataType::FLOAT>(input_shape,
                                                           gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          gpu_allocator.allocate_tensor(replicated_shape);

      Kernels::Replicate::forward_kernel(
          managed_stream.raw_stream(), input_accessor_gpu, output_accessor_gpu);

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_gpu);

      // Run CPU Replicate Forward Kernel
      GenericTensorAccessorW input_accessor_cpu =
          copy_tensor_between_memories<DataType::FLOAT>(input_accessor_gpu,
                                                        cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          cpu_allocator.allocate_tensor(replicated_shape);

      Kernels::Replicate::cpu_forward_kernel(
          read_only_accessor_from_write_accessor(input_accessor_cpu),
          output_accessor_cpu);

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(output_accessor_cpu);

      CHECK(result_data_gpu == result_data_cpu);
    }

    SUBCASE("backward_kernel") {
      // Run GPU Replicate Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r<DataType::FLOAT>(replicated_shape,
                                                           gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          gpu_allocator.allocate_tensor_and_zero(reduced_shape);

      Kernels::Replicate::backward_kernel(managed_stream.raw_stream(),
                                          input_grad_accessor_gpu,
                                          output_grad_accessor_gpu,
                                          num_replicas);

      std::vector<float> result_data_gpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_gpu);

      // Run CPU Replicate Backward Kernel
      GenericTensorAccessorW output_grad_accessor_cpu =
          copy_tensor_between_memories<DataType::FLOAT>(
              output_grad_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          cpu_allocator.allocate_tensor_and_zero(reduced_shape);

      Kernels::Replicate::cpu_backward_kernel(
          input_grad_accessor_cpu,
          read_only_accessor_from_write_accessor(output_grad_accessor_cpu),
          num_replicas);

      std::vector<float> result_data_cpu =
          load_accessor_data<DataType::FLOAT>(input_grad_accessor_cpu);

      CHECK(result_data_gpu == result_data_cpu);
    }
  }
}
