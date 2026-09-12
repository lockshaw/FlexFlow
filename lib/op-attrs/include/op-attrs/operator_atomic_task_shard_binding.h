#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H

#include "op-attrs/tensor_role.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/abstracted_operator_atomic_task_shard_binding.dtg.h"

namespace FlexFlow {

OperatorAtomicTaskShardBinding
  operator_atomic_task_shard_binding_from_abstracted(
    AbstractedOperatorAtomicTaskShardBinding const &);

ParallelTensorSpaceCoordinate
    ptensor_space_coord_for_slot_name(OperatorAtomicTaskShardBinding const &,
                                      TensorSlotName const &);

} // namespace FlexFlow

#endif
