#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H

#include "op-attrs/ops/broadcast_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape broadcast_get_output_shape(BroadcastAttrs const &,
                                                        TensorShape const &);
ParallelTensorDimDegrees broadcast_get_output_parallel_dim_degrees(BroadcastAttrs const &,
                                     ParallelTensorDimDegrees const &);
ParallelTensorShape broadcast_get_output_parallel_shape(BroadcastAttrs const &,
                                     ParallelTensorShape const &);

OperatorTaskSpace broadcast_get_operator_task_space(
    BroadcastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping broadcast_get_operator_to_input_mapping(
    BroadcastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping broadcast_get_operator_to_output_mapping(
    BroadcastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
