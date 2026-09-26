#include "kernels/layer_norm_kernels.h"
#include "kernels/layer_norm_kernels_cpu.h"
#include "kernels/layer_norm_kernels_gpu.h"
#include "utils/optional.h"

namespace FlexFlow {

std::optional<LayerNormPerDeviceState>
    layer_norm_init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                Allocator &allocator,
                bool elementwise_affine,
                int64_t effective_batch_size,
                int64_t effective_num_elements,
                float eps) {
  if (device_type == DeviceType::GPU) {
    return layer_norm_gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*allocator=*/allocator,
        /*elementwise_affine=*/elementwise_affine,
        /*effective_batch_size=*/effective_batch_size,
        /*effective_num_elements=*/effective_num_elements,
        /*eps=*/eps);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void layer_norm_forward_kernel(
    device_stream_t const &stream,
    std::optional<LayerNormPerDeviceState> const &per_device_state,
    LayerNormAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output,
    std::optional<GenericTensorAccessorR> const &gamma,
    std::optional<GenericTensorAccessorR> const &beta) 
{
  if (stream.is_gpu()) {
    layer_norm_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input=*/input,
        /*output=*/output,
        /*gamma=*/assert_unwrap(gamma),
        /*beta=*/assert_unwrap(beta));
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    layer_norm_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output,
        /*gamma=*/gamma,
        /*beta=*/beta);
  }
}

void layer_norm_backward_kernel(
    device_stream_t const &stream,
    std::optional<LayerNormPerDeviceState> const &per_device_state,
    LayerNormAttrs const &attrs,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad) {
  if (stream.is_gpu()) {
    layer_norm_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*gamma=*/gamma,
        /*gamma_grad=*/gamma_grad,
        /*beta_grad=*/beta_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    layer_norm_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*gamma=*/gamma,
        /*gamma_grad=*/gamma_grad,
        /*beta_grad=*/beta_grad);
  }
}

void layer_norm_cleanup_kernel(
    DeviceType device_type,
    std::optional<LayerNormPerDeviceState> const &per_device_state) {
  if (device_type == DeviceType::GPU) {
    layer_norm_gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
