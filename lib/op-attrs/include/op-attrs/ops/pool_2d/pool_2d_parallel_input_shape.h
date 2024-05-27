#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_POOL_2D_POOL_2D_PARALLEL_INPUT_SHAPE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_POOL_2D_POOL_2D_PARALLEL_INPUT_SHAPE_H

#include "op-attrs/ops/pool_2d/pool_2d_parallel_input_shape.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

Pool2DParallelInputShape parse_pool_2d_parallel_input_shape(ParallelTensorShape const &input);

} // namespace FlexFlow

#endif