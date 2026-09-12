#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_ABSTRACTED_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_ABSTRACTED_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H

#include "op-attrs/abstracted_operator_atomic_task_shard_binding.dtg.h"
#include <set>

namespace FlexFlow {

AbstractedOperatorAtomicTaskShardBinding
  restrict_abstracted_operator_atomic_task_shard_binding_to_slots(
    AbstractedOperatorAtomicTaskShardBinding const &,
    std::set<TensorSlotName> const &);

} // namespace FlexFlow

#endif
