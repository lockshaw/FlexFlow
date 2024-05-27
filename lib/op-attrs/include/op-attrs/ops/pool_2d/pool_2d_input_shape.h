#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_POOL_2D_POOL_2D_INPUT_SHAPE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_POOL_2D_POOL_2D_INPUT_SHAPE_H

#include "op-attrs/ops/pool_2d/pool_2d_input_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

Pool2DInputShape parse_pool_2d_input_shape(TensorShape const &input);

} // namespace FlexFlow

#endif