#ifndef _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H

#include "kernels/array_shape.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/operator_type.dtg.h"
#include "kernels/reduce_per_device_state.dtg.h"

namespace FlexFlow {
namespace Kernels {
namespace Reduce {

ReducePerDeviceState init_kernel(PerDeviceFFHandle const &,
                                 OperatorType const &,
                                 size_t const &,
                                 ArrayShape const &input_shape,
                                 ArrayShape const &output_shape);

void forward_kernel(ffStream_t stream,
                    ReducePerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     ReducePerDeviceState const &m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);
} // namespace Reduce
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
