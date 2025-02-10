#ifndef _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H

#include "kernels/device.h"
#include "kernels/allocation.h"
#include "kernels/ff_handle.h"
#include "kernels/layer_norm_per_device_state.dtg.h"

namespace FlexFlow {
namespace Kernels {
namespace LayerNorm {

// todo: this may have some problem.
LayerNormPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                    Allocator &allocator,
                                    bool elementwise_affine,
                                    int64_t effective_batch_size,
                                    int64_t effective_num_elements,
                                    float eps);

void forward_kernel(ffStream_t stream,
                    LayerNormPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorW const &gamma,
                    GenericTensorAccessorW const &beta);

void backward_kernel(ffStream_t stream,
                     LayerNormPerDeviceState const &m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &gamma,
                     GenericTensorAccessorW const &gamma_grad,
                     GenericTensorAccessorW const &beta_grad);

} // namespace LayerNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
