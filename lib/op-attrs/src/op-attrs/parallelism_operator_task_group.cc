#include "op-attrs/parallelism_operator_task_group.h"
#include "op-attrs/abstracted_operator_atomic_task_shard_binding.h"
#include "op-attrs/operator_atomic_task_shard_binding.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/binary_relation/binary_relation.h"
#include "utils/binary_relation/binary_relation_is_left_k_unique.h"
#include "utils/binary_relation/binary_relation_is_right_k_unique.h"
#include "utils/binary_relation/binary_relation_transform_l_and_r.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_element_counts.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/require_two_keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/hash/set.h"
#include "utils/hash/tuple.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/optional.h"

namespace FlexFlow {

ParallelismOperatorTaskGroup::ParallelismOperatorTaskGroup(
    std::set<AbstractedOperatorAtomicTaskShardBinding> const &shard_bindings)
    : bindings(shard_bindings) {

  std::set<TensorSlotName> slot_names = [&]() -> std::set<TensorSlotName> {
    std::vector<std::set<TensorSlotName>> binding_slot_sets = transform(
        vector_of(this->bindings),
        [&](AbstractedOperatorAtomicTaskShardBinding const &s)
            -> std::set<TensorSlotName> { return keys(s.tensor_coords); });

    return require_all_same1(binding_slot_sets);
  }();

  {
    std::set<TensorSlotName> expected_slot_names = {
        TensorSlotName::INPUT,
        TensorSlotName::OUTPUT,
    };
    ASSERT(slot_names == expected_slot_names);
  }

  auto coords_for_slot = [&](TensorSlotName const &slot_name)
      -> std::set<ParallelTensorSpaceCoordinate> {
    return transform(this->bindings,
                     [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                         -> ParallelTensorSpaceCoordinate {
                       return b.tensor_coords.at(slot_name);
                     });
  };

  std::set<TaskSpaceCoordinate> task_space_coordinates =
      transform(this->bindings,
                [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                    -> TaskSpaceCoordinate { return b.task_coord; });

  ASSERT(task_space_coord_set_is_orthotopic(task_space_coordinates));

  ParallelTensorDimDegrees input_dim_degrees =
      assert_unwrap(strict_parallel_tensor_dim_degrees_for_coord_set(
          coords_for_slot(TensorSlotName::INPUT)));
  ParallelTensorDimDegrees output_dim_degrees =
      assert_unwrap(strict_parallel_tensor_dim_degrees_for_coord_set(
          coords_for_slot(TensorSlotName::OUTPUT)));

  std::set<parallel_tensor_dim_idx_t> dim_idxs =
      require_same(get_parallel_tensor_dim_indices(input_dim_degrees),
                   get_parallel_tensor_dim_indices(output_dim_degrees));

  BinaryRelation<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate>
      coord_relation = BinaryRelation<ParallelTensorSpaceCoordinate,
                                      ParallelTensorSpaceCoordinate>{
          transform(shard_bindings,
                    [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                        -> std::pair<ParallelTensorSpaceCoordinate,
                                     ParallelTensorSpaceCoordinate> {
                      auto [input_coord, output_coord] =
                          require_two_keys(b.tensor_coords,
                                           TensorSlotName::INPUT,
                                           TensorSlotName::OUTPUT);
                      return std::pair{
                          input_coord,
                          output_coord,
                      };
                    }),
      };

  std::map<parallel_tensor_dim_idx_t,
           BinaryRelation<nonnegative_int, nonnegative_int>>
      single_dim_coord_relations = generate_map(
          dim_idxs,
          [&](parallel_tensor_dim_idx_t const &dim_idx)
              -> BinaryRelation<nonnegative_int, nonnegative_int> {
            auto component_for_dim =
                [&](ParallelTensorSpaceCoordinate const &coord)
                -> nonnegative_int {
              return ptensor_coord_component_for_ptensor_dim_idx(coord,
                                                                 dim_idx);
            };

            return binary_relation_transform_l_and_r(
                coord_relation, component_for_dim, component_for_dim);
          });

  std::map<parallel_tensor_dim_idx_t, std::optional<nonnegative_int>>
      left_ks_per_dim = map_values(
          single_dim_coord_relations,
          [&](BinaryRelation<nonnegative_int, nonnegative_int> const &rel)
              -> std::optional<nonnegative_int> {
            return binary_relation_is_left_k_unique(rel);
          });

  std::map<parallel_tensor_dim_idx_t, std::optional<nonnegative_int>>
      right_ks_per_dim = map_values(
          single_dim_coord_relations,
          [&](BinaryRelation<nonnegative_int, nonnegative_int> const &rel)
              -> std::optional<nonnegative_int> {
            return binary_relation_is_right_k_unique(rel);
          });

  auto is_k_gt_1 = [&](std::optional<nonnegative_int> const &x) -> bool {
    return (x.has_value() && x.value() > 1);
  };

  auto is_k_eq_1 = [&](std::optional<nonnegative_int> const &x) -> bool {
    return (x.has_value() && x.value() == 1);
  };

  auto get_k_properties_pair =
      [&](std::optional<nonnegative_int> const &x) -> std::pair<bool, bool> {
    return std::pair{
        is_k_gt_1(x),
        is_k_eq_1(x),
    };
  };

  std::multiset<std::optional<nonnegative_int>> left_ks =
      values(left_ks_per_dim);

  std::multiset<std::optional<nonnegative_int>> right_ks =
      values(right_ks_per_dim);

  std::map<std::pair<bool, bool>, positive_int> left_k_property_counts =
      get_element_counts(transform(left_ks, get_k_properties_pair));

  std::map<std::pair<bool, bool>, positive_int> right_k_property_counts =
      get_element_counts(transform(right_ks, get_k_properties_pair));

  std::map<std::pair<bool, bool>, positive_int> expected_k_counts = {
      {
          {true, false},
          1_p,
      },
      {
          {false, true},
          positive_int{num_elements(dim_idxs).int_from_nonnegative_int() - 1},
      },
  };

  ASSERT(left_k_property_counts == expected_k_counts ||
         right_k_property_counts == expected_k_counts);
}

bool ParallelismOperatorTaskGroup::operator==(
    ParallelismOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool ParallelismOperatorTaskGroup::operator!=(
    ParallelismOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool ParallelismOperatorTaskGroup::operator<(
    ParallelismOperatorTaskGroup const &other) const {
  return this->tie() < other.tie();
}

bool ParallelismOperatorTaskGroup::operator>(
    ParallelismOperatorTaskGroup const &other) const {
  return this->tie() > other.tie();
}

bool ParallelismOperatorTaskGroup::operator<=(
    ParallelismOperatorTaskGroup const &other) const {
  return this->tie() <= other.tie();
}

bool ParallelismOperatorTaskGroup::operator>=(
    ParallelismOperatorTaskGroup const &other) const {
  return this->tie() >= other.tie();
}

std::set<AbstractedOperatorAtomicTaskShardBinding> const &
    ParallelismOperatorTaskGroup::get_shard_bindings() const {
  return this->bindings;
}

std::tuple<std::set<AbstractedOperatorAtomicTaskShardBinding> const &>
    ParallelismOperatorTaskGroup::tie() const {
  return std::tie(this->bindings);
}

OperatorTaskSpace task_space_for_parallelism_operator_task_group(
    ParallelismOperatorTaskGroup const &task_group) {
  std::set<TaskSpaceCoordinate> task_space_coords =
      transform(task_group.get_shard_bindings(),
                [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                    -> TaskSpaceCoordinate { return b.task_coord; });

  return assert_unwrap(
      strict_operator_task_space_for_coord_set(task_space_coords));
}

ParallelTensorDimDegrees
    parallel_tensor_space_for_parallelism_operator_task_group_and_slot(
        ParallelismOperatorTaskGroup const &task_group,
        TensorSlotName slot_name) {
  std::set<ParallelTensorSpaceCoordinate> task_space_coords =
      transform(task_group.get_shard_bindings(),
                [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                    -> ParallelTensorSpaceCoordinate {
                  return b.tensor_coords.at(slot_name);
                });

  return assert_unwrap(
      strict_parallel_tensor_dim_degrees_for_coord_set(task_space_coords));
}

ShardSignatureInstance
    shard_signature_instance_from_parallelism_operator_task_group(
        ParallelismOperatorTaskGroup const &task_group) {
  return ShardSignatureInstance{
      transform(task_group.get_shard_bindings(),
                [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                    -> OperatorAtomicTaskShardBinding {
                  return operator_atomic_task_shard_binding_from_abstracted(b);
                }),
  };
}

ParallelismOperatorTaskGroup restrict_parallelism_operator_task_group_to_slots(
    ParallelismOperatorTaskGroup const &task_group,
    std::set<TensorSlotName> const &desired_slots) {
  return ParallelismOperatorTaskGroup{
      /*shard_bindings=*/
      transform(
          task_group.get_shard_bindings(),
          [&](AbstractedOperatorAtomicTaskShardBinding const &b)
              -> AbstractedOperatorAtomicTaskShardBinding {
            return restrict_abstracted_operator_atomic_task_shard_binding_to_slots(
                b, desired_slots);
          }),
  };
}

OperatorAtomicTaskShardBinding
    parallelism_op_task_group_get_binding_for_task_space_coord(
        ParallelismOperatorTaskGroup const &task_group,
        TaskSpaceCoordinate const &task_coord) {
  {
    OperatorTaskSpace task_group_task_space =
        task_space_for_parallelism_operator_task_group(task_group);

    ASSERT(
        operator_task_space_contains_coord(task_group_task_space, task_coord));
  }

  return get_only(
      filtrans(task_group.get_shard_bindings(),
               [&](AbstractedOperatorAtomicTaskShardBinding const &b)
                   -> std::optional<OperatorAtomicTaskShardBinding> {
                 if (b.task_coord == task_coord) {
                   return operator_atomic_task_shard_binding_from_abstracted(b);
                 } else {
                   return std::nullopt;
                 }
               }));
}

} // namespace FlexFlow

///\cond
namespace std {

using ::FlexFlow::ParallelismOperatorTaskGroup;

size_t hash<ParallelismOperatorTaskGroup>::operator()(
    ParallelismOperatorTaskGroup const &t) const {
  return ::FlexFlow::get_std_hash(t.tie());
}

} // namespace std
///\endcond

///\cond
namespace nlohmann {

using ::FlexFlow::ParallelismOperatorTaskGroup;

ParallelismOperatorTaskGroup
    adl_serializer<ParallelismOperatorTaskGroup>::from_json(json const &j) {
  return ParallelismOperatorTaskGroup{
      j.template get<
          ::std::set<::FlexFlow::AbstractedOperatorAtomicTaskShardBinding>>(),
  };
}

void adl_serializer<ParallelismOperatorTaskGroup>::to_json(
    json &j, ParallelismOperatorTaskGroup const &t) {
  j = t.get_shard_bindings();
}

} // namespace nlohmann
///\endcond
