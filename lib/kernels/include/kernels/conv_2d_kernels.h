#ifndef _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H

#include "kernels/device.h"
#include "kernels/accessor.h"
#include "kernels/ff_handle.h"
#include "op-attrs/activation.dtg.h"
#include "kernels/conv_2d_per_device_state.dtg.h"

namespace FlexFlow {
namespace Kernels {
namespace Conv2D {

Conv2DPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 std::optional<Activation> activation,
                                 int kernel_h,
                                 int kernel_w,
                                 int groups,
                                 int padding_h,
                                 int padding_w,
                                 int stride_h,
                                 int stride_w,
                                 GenericTensorAccessorW const &input,
                                 GenericTensorAccessorW const &output,
                                 float const *filter_ptr,
                                 float *filter_grad_ptr);

void forward_kernel(ffStream_t stream,
                    Conv2DPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    std::optional<Activation> activation);

void backward_kernel(ffStream_t stream,
                     Conv2DPerDeviceState const &m,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *filter_ptr,
                     float *filter_grad_ptr,
                     float *bias_grad_ptr,
                     std::optional<Activation> activation);

} // namespace Conv2D
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_CONV_2D_KERNELS_H
