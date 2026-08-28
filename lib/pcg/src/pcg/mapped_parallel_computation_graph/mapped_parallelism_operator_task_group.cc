#include "pcg/mapped_parallel_computation_graph/mapped_parallelism_operator_task_group.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include "utils/containers/vector_of.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "utils/containers/keys.h"
#include "utils/containers/require_all_same.h"
#include "utils/containers/transform_pairs.h"
#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "pcg/mapped_parallel_computation_graph/parallelism_operator_atomic_task_shard_binding.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.h"

namespace FlexFlow {

MappedParallelismOperatorTaskGroup::MappedParallelismOperatorTaskGroup(
    bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> const
        &shard_bindings)
    : shard_bindings(shard_bindings) {

  std::set<MappedOperatorAtomicTaskShardBinding> mapped_bindings =
    transform_pairs(
      unstructured_relation_from_bidict(shard_bindings),
      [&](MachineSpaceCoordinate const &mc, ParallelismOperatorAtomicTaskShardBinding const &b)
        -> MappedOperatorAtomicTaskShardBinding
      {
        return MappedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/
          operator_atomic_task_shard_binding_from_parallelism_op_binding(b).tensor_coords,
          /*machine_coord=*/mc,
        };
      });

  auto satisfies_uniqueness = [&](TensorSlotName slot_1,
                                  TensorSlotName slot_2) 
    -> bool
  {
    std::optional<nonnegative_int> k = 
      mapped_op_task_shard_bindings_are_k_unique_on_slot(mapped_bindings, slot_2);

    return  
      mapped_op_task_shard_bindings_are_unique_on_slot(mapped_bindings, slot_1)
      &&
      k.has_value()
      &&
      k.value() > 1;
  };

  ASSERT(
    satisfies_uniqueness(TensorSlotName::INPUT, TensorSlotName::OUTPUT)
    ||
    satisfies_uniqueness(TensorSlotName::OUTPUT, TensorSlotName::INPUT)
  );
}

bool MappedParallelismOperatorTaskGroup::operator==(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator!=(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() != other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator<(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() < other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator>(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() > other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator<=(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() <= other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator>=(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() >= other.tie();
}

std::tuple<
    bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> const &>
    MappedParallelismOperatorTaskGroup::tie() const {

  return std::tie(this->shard_bindings);
}

bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> const &
    MappedParallelismOperatorTaskGroup::get_shard_bindings() const {
  return this->shard_bindings;
}

nlohmann::json format_as(::FlexFlow::MappedParallelismOperatorTaskGroup const &m) {
  return m;
}

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::MappedParallelismOperatorTaskGroup const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MappedParallelismOperatorTaskGroup>::operator()(
    ::FlexFlow::MappedParallelismOperatorTaskGroup const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std

namespace nlohmann {

::FlexFlow::MappedParallelismOperatorTaskGroup
    adl_serializer<::FlexFlow::MappedParallelismOperatorTaskGroup>::from_json(
        json const &j) {
  return ::FlexFlow::MappedParallelismOperatorTaskGroup{j.template get<
      ::FlexFlow::bidict<::FlexFlow::MachineSpaceCoordinate,
                         ::FlexFlow::ParallelismOperatorAtomicTaskShardBinding>>()};
}

void adl_serializer<::FlexFlow::MappedParallelismOperatorTaskGroup>::to_json(
    json &j, ::FlexFlow::MappedParallelismOperatorTaskGroup const &t) {
  j = t.get_shard_bindings();
}

} // namespace nlohmann
