#include "kernels/element_binary_kernels.h"
#include "kernels/element_binary_kernels_cpu.h"
#include "kernels/element_binary_kernels_gpu.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::optional<ElementBinaryPerDeviceState>
    element_binary_init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                ElementBinaryOp op_type,
                bool should_broadcast_lhs,
                bool should_broadcast_rhs,
                TensorShape const &lhs_shape,
                TensorShape const &rhs_shape,
                TensorShape const &output_shape) {
  if (device_type == DeviceType::GPU) {
    return element_binary_gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*op_type=*/op_type,
        /*should_broadcast_lhs=*/should_broadcast_lhs,
        /*should_broadcast_rhs=*/should_broadcast_rhs,
        /*lhs_shape=*/lhs_shape,
        /*rhs_shape=*/rhs_shape,
        /*output_shape=*/output_shape);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void element_binary_forward_kernel(
    device_stream_t const &stream,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state,
    device_handle_t const &handle,
    ElementBinaryAttrs const &attrs,
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output)
{
  if (stream.is_gpu()) {
    element_binary_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*lhs_ptr=*/lhs.get_float_ptr(),
        /*rhs_ptr=*/rhs.get_float_ptr(),
        /*out_ptr=*/output.get_float_ptr(),
        /*op_type=*/attrs.op,
        /*broadcast_inputLHS=*/attrs.should_broadcast_lhs,
        /*broadcast_inputRHS=*/attrs.should_broadcast_rhs,
        /*handle=*/handle.require_for_gpu());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    ASSERT(handle.is_for_cpu());
    element_binary_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*lhs=*/lhs,
        /*rhs=*/rhs,
        /*out=*/output);
  }
}

void element_binary_backward_kernel(
    device_stream_t const &stream,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state,
    device_handle_t const &handle,
    ElementBinaryAttrs const &attrs,
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorW const &lhs_grad,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &rhs_grad,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad)
{
  if (stream.is_gpu()) {
    element_binary_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*out_grad_ptr=*/output_grad.get_float_ptr(),
        /*lhs_ptr=*/lhs.get_float_ptr(),
        /*rhs_ptr=*/rhs.get_float_ptr(),
        /*lhs_grad_ptr=*/lhs_grad.get_float_ptr(),
        /*rhs_grad_ptr=*/rhs_grad.get_float_ptr(),
        /*op_type=*/attrs.op,
        /*broadcast_inputLHS=*/attrs.should_broadcast_lhs,
        /*broadcast_inputRHS=*/attrs.should_broadcast_rhs,
        /*handle=*/handle.require_for_gpu());
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    ASSERT(handle.is_for_cpu());
    element_binary_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*lhs=*/lhs,
        /*lhs_grad=*/lhs_grad,
        /*rhs=*/rhs,
        /*rhs_grad=*/rhs_grad,
        /*output=*/output,
        /*output_grad=*/output_grad);
  }
}

void element_binary_cleanup_kernel(
    DeviceType device_type,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state) {
  if (device_type == DeviceType::GPU) {
    element_binary_gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
