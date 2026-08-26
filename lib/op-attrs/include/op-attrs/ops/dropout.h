#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_DROPOUT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_DROPOUT_H

#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape dropout_get_output_shape(DropoutAttrs const &, TensorShape const &);

ParallelTensorDimDegrees
    dropout_get_output_parallel_dim_degrees(DropoutAttrs const &, ParallelTensorDimDegrees const &);

ParallelTensorShape
    dropout_get_output_parallel_shape(DropoutAttrs const &, ParallelTensorShape const &);

OperatorTaskSpace dropout_get_operator_task_space(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping dropout_get_operator_to_input_mapping(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping dropout_get_operator_to_output_mapping(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
