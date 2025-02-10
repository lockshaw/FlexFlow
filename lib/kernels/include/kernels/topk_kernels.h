#ifndef _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H

#include "kernels/device.h"
#include "kernels/allocation.h"
#include "kernels/topk_per_device_state.struct.toml"

namespace FlexFlow {

namespace Kernels {
namespace TopK {

TopKPerDeviceState init_kernel(bool sorted);

void forward_kernel(ffStream_t stream,
                    TopKPerDeviceState const &m,
                    float const *input_ptr,
                    float *output_ptr,
                    int *indices_ptr,
                    size_t batch_size,
                    int length,
                    int k,
                    bool sorted);
void backward_kernel(ffStream_t stream,
                     TopKPerDeviceState const &m,
                     float const *out_grad_ptr,
                     int const *indices_ptr,
                     float *in_grad_ptr,
                     size_t batch_size,
                     int length,
                     int k);

} // namespace TopK
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TOPK_KERNELS_H
