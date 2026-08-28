#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_GATHER_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_GATHER_H

#include "op-attrs/ops/gather_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape gather_get_output_shape(GatherAttrs const &,
                             TensorShape const &input,
                             TensorShape const &index);

ParallelTensorDimDegrees gather_get_output_parallel_dim_degrees(
  GatherAttrs const &,
  ParallelTensorDimDegrees const &input,
  ParallelTensorDimDegrees const &index);

ParallelTensorShape gather_get_output_parallel_shape(GatherAttrs const &,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index);

OperatorTaskSpace gather_get_operator_task_space(
    GatherAttrs const &attrs, 
    ParallelTensorDimDegrees const &input_degrees,
    ParallelTensorDimDegrees const &index_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping gather_get_operator_to_input_mapping(
    GatherAttrs const &attrs, 
    ParallelTensorDimDegrees const &input_degrees,
    ParallelTensorDimDegrees const &index_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping gather_get_operator_to_index_mapping(
    GatherAttrs const &attrs, 
    ParallelTensorDimDegrees const &input_degrees,
    ParallelTensorDimDegrees const &index_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping gather_get_operator_to_output_mapping(
    GatherAttrs const &attrs, 
    ParallelTensorDimDegrees const &input_degrees,
    ParallelTensorDimDegrees const &index_degrees);


} // namespace FlexFlow

#endif
