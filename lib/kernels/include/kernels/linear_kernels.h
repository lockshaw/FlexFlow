#ifndef _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LINEAR_KERNELS_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "kernels/linear_per_device_state.dtg.h"

namespace FlexFlow {
namespace Kernels {
namespace Linear {

LinearPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                 float *one_ptr,
                                 std::optional<Activation> activation,
                                 std::optional<RegularizerAttrs> regularizer,
                                 bool use_bias,
                                 DataType input_type,
                                 DataType weight_type,
                                 DataType output_type,
                                 int batch_size,
                                 int channel);

bool use_activation(Activation activation);

void forward_kernel(ffStream_t stream,
                    LinearPerDeviceState const &m,
                    void const *input_ptr,
                    void *output_ptr,
                    void const *filter_ptr,
                    void const *bias_ptr,
                    int in_dim,
                    int out_dim,
                    int batch_size);

void backward_kernel(ffStream_t stream,
                     LinearPerDeviceState const &m,
                     void const *input_ptr,
                     void *input_grad_ptr,
                     void const *output_ptr,
                     void *output_grad_ptr,
                     void const *kernel_ptr,
                     void *kernel_grad_ptr,
                     void *bias_ptr,
                     int in_dim,
                     int out_dim,
                     int batch_size);

} // namespace Linear
} // namespace Kernels
} // namespace FlexFlow

#endif
