#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_FLAT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_FLAT_H

#include "op-attrs/ops/flat_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape flat_get_output_shape(FlatAttrs const &, TensorShape const &);
ParallelTensorDimDegrees
    flat_get_output_parallel_dim_degrees(FlatAttrs const &,
                                         ParallelTensorDimDegrees const &);
ParallelTensorShape flat_get_output_parallel_shape(FlatAttrs const &,
                                                   ParallelTensorShape const &);

OperatorTaskSpace flat_get_operator_task_space(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping flat_get_operator_to_input_mapping(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping flat_get_operator_to_output_mapping(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);


} // namespace FlexFlow

#endif
