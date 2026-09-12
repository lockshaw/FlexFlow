#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/abstracted_operator_atomic_task_shard_binding.h"
#include "op-attrs/operator_atomic_task_shard_binding.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/operator_task_space.h"
#include "utils/optional.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"
#include "utils/orthotope/orthotope_bounded_coord.h"

namespace FlexFlow {

StandardOperatorTaskGroup::StandardOperatorTaskGroup(
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

  ASSERT(task_space_coordinates.size() == shard_bindings.size());
  ASSERT(task_space_coord_set_is_orthotopic(task_space_coordinates));

  for (TensorSlotName slot_name : slot_names) {
    std::set<ParallelTensorSpaceCoordinate> slot_coords = coords_for_slot(slot_name);
    ASSERT(slot_coords.size() == shard_bindings.size());
    ASSERT(parallel_tensor_coord_set_is_orthotopic(slot_coords));
  }
}

bool StandardOperatorTaskGroup::operator==(StandardOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool StandardOperatorTaskGroup::operator!=(StandardOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool StandardOperatorTaskGroup::operator<(StandardOperatorTaskGroup const &other) const {
  return this->tie() < other.tie();
}

bool StandardOperatorTaskGroup::operator>(StandardOperatorTaskGroup const &other) const {
  return this->tie() > other.tie();
}

bool StandardOperatorTaskGroup::operator<=(StandardOperatorTaskGroup const &other) const {
  return this->tie() <= other.tie();
}

bool StandardOperatorTaskGroup::operator>=(StandardOperatorTaskGroup const &other) const {
  return this->tie() >= other.tie();
}

std::set<AbstractedOperatorAtomicTaskShardBinding> const &StandardOperatorTaskGroup::get_shard_bindings() const {
  return this->bindings;
}

std::tuple<
  std::set<AbstractedOperatorAtomicTaskShardBinding> const &
> StandardOperatorTaskGroup::tie() const {
  return std::tie(this->bindings);
}

OperatorTaskSpace 
  task_space_for_standard_operator_task_group(
    StandardOperatorTaskGroup const &task_group)
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
  parallel_tensor_space_for_standard_operator_task_group_and_slot(
    StandardOperatorTaskGroup const &task_group,
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
  shard_signature_instance_from_standard_operator_task_group(
    StandardOperatorTaskGroup const &task_group) 
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

OperatorSpaceToParallelTensorSpaceBiuniqueMapping 
  standard_operator_task_group_get_operator_to_ptensor_mapping(
    StandardOperatorTaskGroup const &task_group,
    TensorSlotName slot_name)
{
  return OperatorSpaceToParallelTensorSpaceBiuniqueMapping{
    DimDomainBiuniqueMapping<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>{
      /*coord_mapping=*/
        bidict_from_unstructured_relation(
          transform(task_group.get_shard_bindings(),
                    [&](AbstractedOperatorAtomicTaskShardBinding const &b) 
                      -> std::pair<
                          DimCoord<operator_task_space_dim_idx_t>,
                          DimCoord<parallel_tensor_dim_idx_t>
                        > 
                    {
                      ParallelTensorSpaceCoordinate pt_coord = b.tensor_coords.at(slot_name);
                      TaskSpaceCoordinate task_coord = b.task_coord;

                      DimCoord<parallel_tensor_dim_idx_t> pt_dim_coord = 
                        dim_coord_from_parallel_tensor_space_coord(pt_coord);
                      DimCoord<operator_task_space_dim_idx_t> task_dim_coord = 
                        dim_coord_from_task_space_coordinate(task_coord); 

                      return std::pair{
                        task_dim_coord,
                        pt_dim_coord,
                      };
                    })),
      /*l_domain=*/lift_minimal_dim_domain(
        minimal_dim_domain_from_operator_task_space(
          task_space_for_standard_operator_task_group(task_group))),
      /*r_domain=*/dim_domain_from_parallel_tensor_dim_degrees(
        parallel_tensor_space_for_standard_operator_task_group_and_slot(task_group, slot_name)),
    },
  };
}

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping 
  standard_operator_task_group_get_ptensor_to_ptensor_mapping(
    StandardOperatorTaskGroup const &task_group,
    TensorSlotName lhs_slot_name,
    TensorSlotName rhs_slot_name)
{
  return ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping{
    DimDomainBiuniqueMapping<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>{
      /*coord_mapping=*/
        bidict_from_unstructured_relation(
          transform(task_group.get_shard_bindings(),
                    [&](AbstractedOperatorAtomicTaskShardBinding const &b) 
                      -> std::pair<
                          DimCoord<parallel_tensor_dim_idx_t>,
                          DimCoord<parallel_tensor_dim_idx_t>
                        > 
                    {
                      ParallelTensorSpaceCoordinate left_pt_coord = b.tensor_coords.at(lhs_slot_name);
                      ParallelTensorSpaceCoordinate right_pt_coord = b.tensor_coords.at(rhs_slot_name);

                      DimCoord<parallel_tensor_dim_idx_t> left_pt_dim_coord = 
                        dim_coord_from_parallel_tensor_space_coord(left_pt_coord);
                      DimCoord<parallel_tensor_dim_idx_t> right_pt_dim_coord = 
                        dim_coord_from_parallel_tensor_space_coord(right_pt_coord); 

                      return std::pair{
                        left_pt_dim_coord,
                        right_pt_dim_coord,
                      };
                    })),
      /*l_domain=*/dim_domain_from_parallel_tensor_dim_degrees(
        parallel_tensor_space_for_standard_operator_task_group_and_slot(task_group, lhs_slot_name)),
      /*r_domain=*/dim_domain_from_parallel_tensor_dim_degrees(
        parallel_tensor_space_for_standard_operator_task_group_and_slot(task_group, rhs_slot_name)),
    },
  };
}

StandardOperatorTaskGroup
  standard_operator_task_group_from_shard_signature_instance(
    ShardSignatureInstance const &shard_signature_instance,
    std::function<TaskSpaceCoordinate(OperatorAtomicTaskShardBinding const &)> const &get_task_space_coord)
{
  return StandardOperatorTaskGroup{
    transform(
      shard_signature_instance.get_shard_bindings(),
      [&](OperatorAtomicTaskShardBinding const &b)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/b.tensor_coords,
          /*task_coord=*/get_task_space_coord(b),
        };
      }),
  };
}

StandardOperatorTaskGroup
  restrict_standard_operator_task_group_to_slots(
    StandardOperatorTaskGroup const &task_group,
    std::set<TensorSlotName> const &desired_slots)
{
  return StandardOperatorTaskGroup{
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

} // namespace FlexFlow
