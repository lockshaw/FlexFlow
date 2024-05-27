#ifndef _FLEXFLOW_POOL_2D_ATTRS_H
#define _FLEXFLOW_POOL_2D_ATTRS_H

#include "core.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(Pool2DAttrs);

TensorShape get_kernel_shape(Pool2DAttrs const &attrs, TensorShape const &input);
TensorShape get_output_shape(Pool2DAttrs const &attrs, TensorShape const &input);

ParallelTensorShape get_kernel_shape(Pool2DAttrs const &attrs, ParallelTensorShape const &input_shape);
ParallelTensorShape get_output_shape(Pool2DAttrs const &attrs, ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
