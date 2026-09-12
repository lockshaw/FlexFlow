#include "op-attrs/abstracted_operator_atomic_task_shard_binding.h"
#include "utils/containers/restrict_keys_strict.h"

namespace FlexFlow {

AbstractedOperatorAtomicTaskShardBinding
  restrict_abstracted_operator_atomic_task_shard_binding_to_slots(
    AbstractedOperatorAtomicTaskShardBinding const &binding,
    std::set<TensorSlotName> const &desired_slots)
{
  return AbstractedOperatorAtomicTaskShardBinding{
    /*tensor_coords=*/restrict_keys_strict(binding.tensor_coords, desired_slots),
    /*task_coord=*/binding.task_coord,
  };
}

} // namespace FlexFlow
