#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCTION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCTION_H

#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"

namespace FlexFlow {

ParallelTensorShape
    reduction_get_output_parallel_shape(ReductionAttrs const &attrs,
                     ParallelTensorShape const &input_shape);

ParallelTensorDimDegrees reduction_get_output_parallel_dim_degrees(
    ReductionAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace
    reduction_get_operator_task_space(ReductionAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    reduction_get_operator_to_input_mapping(
        ReductionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    reduction_get_operator_to_output_mapping(
        ReductionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
