#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.h"
#include "utils/containers/keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/transform.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/binary_relation/binary_relation.h"
#include "utils/binary_relation/binary_relation_is_right_k_unique.h"
#include "utils/binary_relation/binary_relation_is_biunique.h"
#include "utils/binary_relation/binary_relation_is_left_k_unique.h"

namespace FlexFlow {

std::pair<ParallelTensorSpaceCoordinate, MappedOperatorAtomicTaskShardBinding>
  mapped_op_task_shard_binding_project_out_key(MappedOperatorAtomicTaskShardBinding const &b, TensorSlotName key)
{
  std::set<TensorSlotName> remaining_keys =
    set_minus(keys(b.tensor_coords), std::set{key});

  return std::pair{
    b.tensor_coords.at(key),
    MappedOperatorAtomicTaskShardBinding{
      /*tensor_coords=*/restrict_keys_strict(b.tensor_coords, remaining_keys),
      /*machine_coord=*/b.machine_coord,
    },
  };
}

std::optional<nonnegative_int>
  mapped_op_task_shard_bindings_are_k_unique_on_slot(
    std::set<MappedOperatorAtomicTaskShardBinding> const &bindings,
    TensorSlotName slot_name)
{
  std::set<std::pair<
    ParallelTensorSpaceCoordinate,
    MappedOperatorAtomicTaskShardBinding
  >> s = transform(bindings,
                  [&](MappedOperatorAtomicTaskShardBinding const &b)
                    -> std::pair<ParallelTensorSpaceCoordinate, MappedOperatorAtomicTaskShardBinding>
                  {
                    return mapped_op_task_shard_binding_project_out_key(b, slot_name);
                  });

  BinaryRelation<
    ParallelTensorSpaceCoordinate,
    MappedOperatorAtomicTaskShardBinding
  > r = BinaryRelation{s};

  return binary_relation_is_right_k_unique(r);
}

std::optional<int_ge_two>
  mapped_op_task_shard_bindings_are_strictly_k_unique_on_slot(
    std::set<MappedOperatorAtomicTaskShardBinding> const &bindings,
    TensorSlotName slot_name) 
{
  std::optional<nonnegative_int> k =
    mapped_op_task_shard_bindings_are_k_unique_on_slot(bindings, slot_name);

  if (!k.has_value()) {
    return std::nullopt;
  }

  if (k.value() <= 1) {
    return std::nullopt;
  }

  return int_ge_two{k.value()};
}

bool
  mapped_op_task_shard_bindings_are_unique_on_slot(
    std::set<MappedOperatorAtomicTaskShardBinding> const &bindings,
    TensorSlotName slot_name)
{
  std::set<std::pair<
    ParallelTensorSpaceCoordinate,
    MappedOperatorAtomicTaskShardBinding
  >> s = transform(bindings,
                  [&](MappedOperatorAtomicTaskShardBinding const &b)
                    -> std::pair<ParallelTensorSpaceCoordinate, MappedOperatorAtomicTaskShardBinding>
                  {
                    return mapped_op_task_shard_binding_project_out_key(b, slot_name);
                  });

  BinaryRelation<
    ParallelTensorSpaceCoordinate,
    MappedOperatorAtomicTaskShardBinding
  > r = BinaryRelation{s};

  return binary_relation_is_biunique(r);
}

} // namespace FlexFlow
