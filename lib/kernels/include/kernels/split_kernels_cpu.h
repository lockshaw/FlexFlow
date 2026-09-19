#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SPLIT_KERNELS_CPU_H

namespace FlexFlow {

void split_cpu_forward_kernel(SplitAttrs const &attrs,
                              GenericTensorAccessorR const &,
                              std::vector<GenericTensorAccessorW> const &outputs);

void split_cpu_backward_kernel(SplitAttrs const &attrs,
                               std::vector<GenericTensorAccessorR> const &output_grads,
                               GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
