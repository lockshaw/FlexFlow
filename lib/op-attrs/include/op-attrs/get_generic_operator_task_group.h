#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_GENERIC_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_GENERIC_OPERATOR_TASK_GROUP_H

#include "op-attrs/generic_operator_task_group.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"

namespace FlexFlow {

GenericOperatorTaskGroup get_generic_operator_task_group(
    PCGOperatorAttrs const &,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &input_dim_degrees);

} // namespace FlexFlow

#endif
