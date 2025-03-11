#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_BINARY_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/array_shape.h"
#include "op-attrs/datatype.h"
#include "op-attrs/operator_type.h"
#include "kernels/element_binary_per_device_state.dtg.h"

namespace FlexFlow::Kernels::ElementBinary {

ElementBinaryPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                        OperatorType op_type,
                                        bool should_broadcast_lhs,
                                        bool should_broadcast_rhs,
                                        ArrayShape lhs_shape,
                                        ArrayShape rhs_shape,
                                        ArrayShape output_shape);

void forward_kernel(ffStream_t stream,
                    ElementBinaryPerDeviceState const &m,
                    float const *lhs_ptr,
                    float const *rhs_ptr,
                    float *out_ptr,
                    OperatorType op_type,
                    bool broadcast_inputLHS,
                    PerDeviceFFHandle handle);

void backward_kernel(ffStream_t stream,
                     ElementBinaryPerDeviceState const &m,
                     float const *out_grad_ptr,
                     float const *lhs_ptr,
                     float const *rhs_ptr,
                     float *lhs_grad_ptr,
                     float *rhs_grad_ptr,
                     OperatorType op_type,
                     bool broadcast_inputLHS,
                     bool broadcast_inputRHS,
                     PerDeviceFFHandle handle);

} // namespace FlexFlow::Kernels::ElementBinary

#endif
