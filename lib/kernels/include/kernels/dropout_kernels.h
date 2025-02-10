#ifndef _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H

#include "kernels/device.h"
#include "kernels/allocation.h"
#include "kernels/array_shape.h"
#include "kernels/ff_handle.h"
#include "kernels/dropout_per_device_state.dtg.h"
#include <cstddef>

namespace FlexFlow::Kernels::Dropout {

DropoutPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                  float rate,
                                  unsigned long long seed,
                                  ArrayShape const &output_domain,
                                  Allocator allocator);

void forward_kernel(ffStream_t stream,
                    DropoutPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     DropoutPerDeviceState const &m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);

void cleanup_kernel(Allocator allocator,
                    ffTensorDescriptor_t inputTensor,
                    ffTensorDescriptor_t outputTensor,
                    ffDropoutDescriptor_t dropoutDesc,
                    void *dropoutStates);

} // namespace FlexFlow::Kernels::Dropout

#endif // _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
