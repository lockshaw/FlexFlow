#ifndef _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"
#include "kernels/repartition_per_device_state.dtg.h"

namespace FlexFlow::Kernels::Repartition {

RepartitionPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                      DataType data_type);

void forward_kernel(ffStream_t stream,
                    RepartitionPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     RepartitionPerDeviceState const &m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow::Kernels::Repartition

#endif // _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
