#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_CPU_H

namespace FlexFlow {

void pool2d_cpu_forward_kernel(Pool2DAttrs const &attrs,
                               GenericTensorAccessorR const &input,
                               GenericTensorAccessorW const &output);

void pool2d_cpu_backward_kernel(Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &output_grad,
                                GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow::Kernels::Pool2D

#endif
