#ifndef _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SOFTMAX_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/softmax_per_device_state.dtg.h"

namespace FlexFlow {

namespace Kernels {
namespace Softmax {

SoftmaxPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                  int dim,
                                  int input_n,
                                  int input_c,
                                  int input_h,
                                  int input_w);

void forward_kernel(ffStream_t stream,
                    SoftmaxPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements);

} // namespace Softmax
} // namespace Kernels
} // namespace FlexFlow

#endif
