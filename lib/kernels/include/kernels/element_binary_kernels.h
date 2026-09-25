#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/element_binary_per_device_state.dtg.h"
#include "op-attrs/operator_type.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_type.dtg.h"
#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "kernels/accessor.h"

namespace FlexFlow {

std::optional<ElementBinaryPerDeviceState>
    element_binary_init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                ElementBinaryOp op_type,
                bool should_broadcast_lhs,
                bool should_broadcast_rhs,
                TensorShape const &lhs_shape,
                TensorShape const &rhs_shape,
                TensorShape const &output_shape);

void element_binary_forward_kernel(
    device_stream_t const &stream,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state,
    device_handle_t const &handle,
    ElementBinaryAttrs const &attrs,
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output);

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
    GenericTensorAccessorR const &output_grad);

void element_binary_cleanup_kernel(
    DeviceType device_type,
    std::optional<ElementBinaryPerDeviceState> const &per_device_state);

} // namespace FlexFlow

#endif
