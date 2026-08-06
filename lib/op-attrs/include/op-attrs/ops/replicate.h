#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

ParallelTensorShape replicate_get_output_parallel_shape(ReplicateAttrs const &attrs,
                                               ParallelTensorShape const &input_shape);

ParallelTensorDimDegrees replicate_get_output_parallel_dim_degrees(
    ReplicateAttrs const &, ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace replicate_get_operator_task_space(
    ReplicateAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    replicate_get_operator_to_input_mapping(
        ReplicateAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    replicate_get_operator_to_output_mapping(
        ReplicateAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
