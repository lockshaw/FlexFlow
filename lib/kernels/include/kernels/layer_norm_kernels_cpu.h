#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LAYER_NORM_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LAYER_NORM_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"

namespace FlexFlow {

void layer_norm_cpu_forward_kernel(LayerNormAttrs const &attrs,
                                   GenericTensorAccessorR const &input,
                                   GenericTensorAccessorW const &output,
                                   std::optional<GenericTensorAccessorR> const &gamma,
                                   std::optional<GenericTensorAccessorR> const &beta);

void layer_norm_cpu_backward_kernel(LayerNormAttrs const &attrs,
                                    GenericTensorAccessorR const &output_grad,
                                    GenericTensorAccessorR const &input,
                                    GenericTensorAccessorW const &input_grad,
                                    GenericTensorAccessorR const &gamma,
                                    GenericTensorAccessorW const &gamma_grad,
                                    GenericTensorAccessorW const &beta_grad);

} // namespace FlexFlow

#endif
