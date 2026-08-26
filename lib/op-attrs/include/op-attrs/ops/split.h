#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SPLIT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SPLIT_H

#include "op-attrs/ops/split_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <vector>
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

std::vector<TensorShape> split_get_output_shapes(SplitAttrs const &,
                                           TensorShape const &);

std::vector<ParallelTensorDimDegrees>
    split_get_output_parallel_dim_degrees(SplitAttrs const &attrs,
                                    ParallelTensorDimDegrees const &input);

std::vector<ParallelTensorShape>
    split_get_output_parallel_shapes(SplitAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

OperatorTaskSpace split_get_operator_task_space(
    SplitAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping split_get_operator_to_input_mapping(
    SplitAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

std::vector<OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
  split_get_operator_to_output_mappings(
    SplitAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
