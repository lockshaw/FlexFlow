#ifndef _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_REDUCE_ATTRS_H

#include "op-attrs/ops/reduce_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReduceAttrs const &,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
