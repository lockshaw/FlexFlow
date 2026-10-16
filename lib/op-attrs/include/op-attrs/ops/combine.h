#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_COMBINE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_COMBINE_H

#include "op-attrs/ops/combine_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/parallelism_operator_task_group.h"

namespace FlexFlow {

ParallelTensorShape
    combine_get_output_parallel_shape(CombineAttrs const &,
                                      ParallelTensorShape const &);

ParallelTensorDimDegrees combine_get_output_parallel_dim_degrees(
    CombineAttrs const &, ParallelTensorDimDegrees const &input_degrees);

ParallelismOperatorTaskGroup
    combine_get_task_group(CombineAttrs const &,
                           ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
