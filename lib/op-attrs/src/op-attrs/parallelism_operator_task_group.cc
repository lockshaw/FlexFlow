#include "op-attrs/parallelism_operator_task_group.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/optional.h"
#include "utils/containers/transform.h"
#include "op-attrs/abstracted_operator_atomic_task_shard_binding.h"
#include "op-attrs/operator_atomic_task_shard_binding.h"
#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

ParallelismOperatorTaskGroup::ParallelismOperatorTaskGroup(
    std::set<AbstractedOperatorAtomicTaskShardBinding> const &shard_bindings)
  : bindings(shard_bindings)
{
  std::set<TensorSlotName> slot_names = [&]() -> std::set<TensorSlotName> {
    std::vector<std::set<TensorSlotName>> binding_slot_sets = transform(
        vector_of(this->bindings),
        [&](AbstractedOperatorAtomicTaskShardBinding const &s) -> std::set<TensorSlotName> {
          return keys(s.tensor_coords);
        });

    return require_all_same1(binding_slot_sets);
  }();

  auto coords_for_slot = [&](TensorSlotName const &slot_name)
    -> std::set<ParallelTensorSpaceCoordinate>
  {
    return transform(
      this->bindings,
      [&](AbstractedOperatorAtomicTaskShardBinding const &b)
        -> ParallelTensorSpaceCoordinate
      {
        return b.tensor_coords.at(slot_name);
      });
  };

  std::set<TaskSpaceCoordinate> task_space_coordinates =
    transform(
      this->bindings,
      [&](AbstractedOperatorAtomicTaskShardBinding const &b)
        -> TaskSpaceCoordinate
      {
        return b.task_coord;
      });

  ASSERT(task_space_coord_set_is_orthotopic(task_space_coordinates));

  // TODO(@lockshaw)(#pr): check that the bindings are k-hemiunique in one dimension

  for (TensorSlotName slot_name : slot_names) {
    std::set<ParallelTensorSpaceCoordinate> slot_coords = coords_for_slot(slot_name);
    ASSERT(parallel_tensor_coord_set_is_orthotopic(slot_coords));
  }
}

bool ParallelismOperatorTaskGroup::operator==(ParallelismOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool ParallelismOperatorTaskGroup::operator!=(ParallelismOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool ParallelismOperatorTaskGroup::operator<(ParallelismOperatorTaskGroup const &other) const {
  return this->tie() < other.tie();
}

bool ParallelismOperatorTaskGroup::operator>(ParallelismOperatorTaskGroup const &other) const {
  return this->tie() > other.tie();
}

bool ParallelismOperatorTaskGroup::operator<=(ParallelismOperatorTaskGroup const &other) const {
  return this->tie() <= other.tie();
}

bool ParallelismOperatorTaskGroup::operator>=(ParallelismOperatorTaskGroup const &other) const {
  return this->tie() >= other.tie();
}

std::set<AbstractedOperatorAtomicTaskShardBinding> const &ParallelismOperatorTaskGroup::get_shard_bindings() const {
  return this->bindings;
}

std::tuple<
  std::set<AbstractedOperatorAtomicTaskShardBinding> const &
> ParallelismOperatorTaskGroup::tie() const {
  return std::tie(this->bindings);
}

OperatorTaskSpace
  task_space_for_parallelism_operator_task_group(
    ParallelismOperatorTaskGroup const &task_group)
{
  std::set<TaskSpaceCoordinate> task_space_coords =
    transform(task_group.get_shard_bindings(),
              [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                -> TaskSpaceCoordinate
              {
                return b.task_coord;
              });

  return assert_unwrap(strict_operator_task_space_for_coord_set(task_space_coords));
}

ParallelTensorDimDegrees
  parallel_tensor_space_for_parallelism_operator_task_group_and_slot(
    ParallelismOperatorTaskGroup const &task_group,
    TensorSlotName slot_name)
{
  std::set<ParallelTensorSpaceCoordinate> task_space_coords =
    transform(task_group.get_shard_bindings(),
              [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                -> ParallelTensorSpaceCoordinate
              {
                return b.tensor_coords.at(slot_name);
              });

  return assert_unwrap(strict_parallel_tensor_dim_degrees_for_coord_set(task_space_coords));
}

ShardSignatureInstance
  shard_signature_instance_from_parallelism_operator_task_group(
    ParallelismOperatorTaskGroup const &task_group)
{
  return ShardSignatureInstance{
    transform(task_group.get_shard_bindings(),
              [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                -> OperatorAtomicTaskShardBinding
              {
                return operator_atomic_task_shard_binding_from_abstracted(b);
              }),
  };
}

ParallelismOperatorTaskGroup
  restrict_parallelism_operator_task_group_to_slots(
    ParallelismOperatorTaskGroup const &task_group,
    std::set<TensorSlotName> const &desired_slots)
{
  return ParallelismOperatorTaskGroup{
    /*shard_bindings=*/
      transform(
        task_group.get_shard_bindings(),
        [&](AbstractedOperatorAtomicTaskShardBinding const &b)
          -> AbstractedOperatorAtomicTaskShardBinding
        {
          return restrict_abstracted_operator_atomic_task_shard_binding_to_slots(b, desired_slots);
        }),
  };
}

OperatorAtomicTaskShardBinding
  parallelism_op_task_group_get_binding_for_task_space_coord(ParallelismOperatorTaskGroup const &task_group,
                                                             TaskSpaceCoordinate const &task_coord)
{
  {
    OperatorTaskSpace task_group_task_space = 
      task_space_for_parallelism_operator_task_group(task_group);

    ASSERT(operator_task_space_contains_coord(task_group_task_space, task_coord));
  }

  return get_only(
    filtrans(task_group.get_shard_bindings(),
           [&](AbstractedOperatorAtomicTaskShardBinding const &b)
             -> std::optional<OperatorAtomicTaskShardBinding>
           {
             if (b.task_coord == task_coord) {
               return operator_atomic_task_shard_binding_from_abstracted(b);
             } else {
               return std::nullopt;
             }
           }));
}

} // namespace FlexFlow

namespace std {

using ::FlexFlow::ParallelismOperatorTaskGroup;

size_t hash<ParallelismOperatorTaskGroup>::operator()(ParallelismOperatorTaskGroup const &t) const {
  return ::FlexFlow::get_std_hash(t.tie());
}

}

namespace nlohmann {

using ::FlexFlow::ParallelismOperatorTaskGroup;

ParallelismOperatorTaskGroup adl_serializer<ParallelismOperatorTaskGroup>::from_json(json const &j) {
  return ParallelismOperatorTaskGroup{
    j.template get<
      ::std::set<::FlexFlow::AbstractedOperatorAtomicTaskShardBinding>
      >(),
  };
}

void adl_serializer<ParallelismOperatorTaskGroup>::to_json(json &j, ParallelismOperatorTaskGroup const &t) {
  j = t.get_shard_bindings();
}

} // namespace nlohmann
