#include "pcg/mapped_parallel_computation_graph/mapped_standard_operator_task_group.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/require_all_same.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "utils/containers/transform_pairs.h"
#include "utils/binary_relation/binary_relation_is_biunique.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.dtg.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

MappedStandardOperatorTaskGroup::MappedStandardOperatorTaskGroup(
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const
        &shard_bindings)
    : shard_bindings(shard_bindings) {
  std::vector<std::set<TensorSlotName>> binding_slot_sets = transform(
      vector_of(shard_bindings.right_values()),
      [&](OperatorAtomicTaskShardBinding const &s) -> std::set<TensorSlotName> {
        return keys(s.tensor_coords);
      });

  std::set<TensorSlotName> slot_names =
      require_all_same(binding_slot_sets).value();

  auto project_out_key = [&](MappedOperatorAtomicTaskShardBinding const &b, TensorSlotName key)
    -> std::pair<
        ParallelTensorSpaceCoordinate,
        MappedOperatorAtomicTaskShardBinding
       >
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
  };

  std::set<MappedOperatorAtomicTaskShardBinding> mapped_bindings =
    transform_pairs(
      unstructured_relation_from_bidict(shard_bindings),
      [&](MachineSpaceCoordinate const &mc, OperatorAtomicTaskShardBinding const &b)
        -> MappedOperatorAtomicTaskShardBinding
      {
        return MappedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/b.tensor_coords,
          /*machine_coord=*/mc,
        };
      });

  auto is_1_unique = [&](TensorSlotName slot_name)
    -> bool
  {
    std::set<std::pair<
      ParallelTensorSpaceCoordinate,
      MappedOperatorAtomicTaskShardBinding
    >> s = transform(mapped_bindings,
                    [&](MappedOperatorAtomicTaskShardBinding const &b)
                      -> std::pair<ParallelTensorSpaceCoordinate, MappedOperatorAtomicTaskShardBinding>
                    {
                      return project_out_key(b, slot_name);
                    });

    BinaryRelation<
      ParallelTensorSpaceCoordinate,
      MappedOperatorAtomicTaskShardBinding
    > r = BinaryRelation{s};

    return binary_relation_is_biunique(r);
  };

  for (TensorSlotName const &slot_name : slot_names) {
    ASSERT(is_1_unique(slot_name));

    std::vector<OperatorAtomicTaskShardBinding> signatures_for_key =
        vector_of(shard_bindings.right_values());

    std::vector<ParallelTensorSpaceCoordinate> coords_for_key = transform(
        signatures_for_key,
        [&](OperatorAtomicTaskShardBinding const &signature) {
          return ptensor_space_coord_for_slot_name(signature, slot_name);
        });

    std::vector<num_ptensor_parallel_dims_t> coord_dims_for_key =
        transform(coords_for_key, [](ParallelTensorSpaceCoordinate const &c) {
          return ptensor_coord_num_dims(c);
        });

    require_all_same(coord_dims_for_key);
  }
}

bool MappedStandardOperatorTaskGroup::operator==(
    MappedStandardOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool MappedStandardOperatorTaskGroup::operator!=(
    MappedStandardOperatorTaskGroup const &other) const {
  return this->tie() != other.tie();
}

bool MappedStandardOperatorTaskGroup::operator<(
    MappedStandardOperatorTaskGroup const &other) const {
  return this->tie() < other.tie();
}

bool MappedStandardOperatorTaskGroup::operator>(
    MappedStandardOperatorTaskGroup const &other) const {
  return this->tie() > other.tie();
}

bool MappedStandardOperatorTaskGroup::operator<=(
    MappedStandardOperatorTaskGroup const &other) const {
  return this->tie() <= other.tie();
}

bool MappedStandardOperatorTaskGroup::operator>=(
    MappedStandardOperatorTaskGroup const &other) const {
  return this->tie() >= other.tie();
}

std::tuple<
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &>
    MappedStandardOperatorTaskGroup::tie() const {

  return std::tie(this->shard_bindings);
}

bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const &
    MappedStandardOperatorTaskGroup::get_shard_bindings() const {
  return this->shard_bindings;
}

nlohmann::json format_as(::FlexFlow::MappedStandardOperatorTaskGroup const &m) {
  return m;
}

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::MappedStandardOperatorTaskGroup const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MappedStandardOperatorTaskGroup>::operator()(
    ::FlexFlow::MappedStandardOperatorTaskGroup const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std

namespace nlohmann {

::FlexFlow::MappedStandardOperatorTaskGroup
    adl_serializer<::FlexFlow::MappedStandardOperatorTaskGroup>::from_json(
        json const &j) {
  return ::FlexFlow::MappedStandardOperatorTaskGroup{j.template get<
      ::FlexFlow::bidict<::FlexFlow::MachineSpaceCoordinate,
                         ::FlexFlow::OperatorAtomicTaskShardBinding>>()};
}

void adl_serializer<::FlexFlow::MappedStandardOperatorTaskGroup>::to_json(
    json &j, ::FlexFlow::MappedStandardOperatorTaskGroup const &t) {
  j = t.get_shard_bindings();
}

} // namespace nlohmann
