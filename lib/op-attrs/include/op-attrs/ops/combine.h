#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_COMBINE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_COMBINE_H

#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/combine_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

ParallelTensorShape
    combine_get_output_parallel_shape(CombineAttrs const &,
                                      ParallelTensorShape const &);

ParallelTensorDimDegrees combine_get_output_parallel_dim_degrees(
    CombineAttrs const &, ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace
    combine_get_operator_task_space(CombineAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
