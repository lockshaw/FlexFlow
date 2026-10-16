#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BROADCAST_H

#include "op-attrs/ops/broadcast_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape broadcast_get_output_shape(BroadcastAttrs const &,
                                       TensorShape const &);
ParallelTensorDimDegrees
    broadcast_get_output_parallel_dim_degrees(BroadcastAttrs const &,
                                              ParallelTensorDimDegrees const &);
ParallelTensorShape
    broadcast_get_output_parallel_shape(BroadcastAttrs const &,
                                        ParallelTensorShape const &);

StandardOperatorTaskGroup
    broadcast_get_task_group(BroadcastAttrs const &,
                             ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
