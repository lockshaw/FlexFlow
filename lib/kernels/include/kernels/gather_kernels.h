#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/gather_per_device_state.dtg.h"

namespace FlexFlow {
namespace Kernels {
namespace Gather {

void forward_kernel(ffStream_t stream,
                    GatherPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     GatherPerDeviceState const &m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad);

} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

#endif
