#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H

#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape reverse_get_output_shape(ReverseAttrs const &, TensorShape const &);
ParallelTensorDimDegrees
    reverse_get_output_parallel_dim_degrees(ReverseAttrs const &,
                                            ParallelTensorDimDegrees const &);
ParallelTensorShape reverse_get_output_parallel_shape(ReverseAttrs const &attrs,
                                                      ParallelTensorShape const &input_shape);

OperatorTaskSpace reverse_get_operator_task_space(
    ReverseAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reverse_get_operator_to_input_mapping(
    ReverseAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reverse_get_operator_to_output_mapping(
    ReverseAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
