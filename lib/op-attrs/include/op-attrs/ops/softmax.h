#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_H

#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape
    softmax_get_output_shape(SoftmaxAttrs const &attrs, TensorShape const &input_shape);

ParallelTensorDimDegrees
    softmax_get_output_parallel_dim_degrees(SoftmaxAttrs const &attrs,
                     ParallelTensorDimDegrees const &input_shape);

ParallelTensorShape
    softmax_get_output_parallel_shape(SoftmaxAttrs const &attrs,
                     ParallelTensorShape const &input_shape);

OperatorTaskSpace softmax_get_operator_task_space(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping softmax_get_operator_to_input_mapping(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping softmax_get_operator_to_output_mapping(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
