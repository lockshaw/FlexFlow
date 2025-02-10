#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "kernels/device.h"
#include "kernels/accessor.h"
#include "kernels/reshape_per_device_state.dtg.h"

namespace FlexFlow {

namespace Kernels {
namespace Reshape {

ReshapePerDeviceState init_kernel(DataType data_type);

void forward_kernel(ffStream_t stream,
                    ReshapePerDeviceState const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     ReshapePerDeviceState const &per_device_state,
                     GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output);

} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
