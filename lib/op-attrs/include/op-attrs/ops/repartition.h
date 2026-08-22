#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

ParallelTensorShape repartition_get_output_parallel_shape(
    RepartitionAttrs const &, ParallelTensorShape const &input_shape);

ParallelTensorDimDegrees repartition_get_output_parallel_dim_degrees(
    RepartitionAttrs const &, ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace repartition_get_operator_task_space(
    RepartitionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    repartition_get_operator_to_input_mapping(
        RepartitionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    repartition_get_operator_to_output_mapping(
        RepartitionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
