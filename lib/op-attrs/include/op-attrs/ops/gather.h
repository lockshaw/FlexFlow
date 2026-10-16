#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_GATHER_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_GATHER_H

#include "op-attrs/ops/gather_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

TensorShape gather_get_output_shape(GatherAttrs const &,
                                    TensorShape const &input,
                                    TensorShape const &index);

ParallelTensorDimDegrees gather_get_output_parallel_dim_degrees(
    GatherAttrs const &,
    ParallelTensorDimDegrees const &input,
    ParallelTensorDimDegrees const &index);

ParallelTensorShape
    gather_get_output_parallel_shape(GatherAttrs const &,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index);

StandardOperatorTaskGroup
    gather_get_task_group(GatherAttrs const &,
                          ParallelTensorDimDegrees const &input,
                          ParallelTensorDimDegrees const &index);

} // namespace FlexFlow

#endif
