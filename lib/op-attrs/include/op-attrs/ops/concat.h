#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_H

#include "op-attrs/ops/concat_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"

namespace FlexFlow {

TensorShape concat_get_output_shape(ConcatAttrs const &,
                                    std::vector<TensorShape> const &);

ParallelTensorDimDegrees
    concat_get_output_parallel_dim_degrees(ConcatAttrs const &,
                                           std::vector<ParallelTensorDimDegrees> const &);

ParallelTensorShape
    concat_get_output_parallel_shape(ConcatAttrs const &,
                                     std::vector<ParallelTensorShape> const &);

OperatorTaskSpace concat_get_operator_task_space(
    ConcatAttrs const &attrs,
    std::vector<ParallelTensorDimDegrees> const &);

std::vector<OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
  concat_get_operator_to_input_mappings(
    ConcatAttrs const &attrs,
    std::vector<ParallelTensorDimDegrees> const &);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  concat_get_operator_to_output_mapping(
    ConcatAttrs const &attrs,
    std::vector<ParallelTensorDimDegrees> const &);

} // namespace FlexFlow

#endif
