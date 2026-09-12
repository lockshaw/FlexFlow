#include "op-attrs/shard_signature_instance.h"
#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/optional.h"

namespace FlexFlow {

ShardSignatureInstance::ShardSignatureInstance(
      std::set<OperatorAtomicTaskShardBinding> const &shard_bindings)
  : bindings(shard_bindings)
{
  // TODO(@lockshaw)(#pr): 
}

bool ShardSignatureInstance::operator==(ShardSignatureInstance const &other) const {
  return this->tie() == other.tie();
}

bool ShardSignatureInstance::operator!=(ShardSignatureInstance const &other) const {
  return this->tie() != other.tie();
}

bool ShardSignatureInstance::operator<(ShardSignatureInstance const &other) const {
  return this->tie() < other.tie();
}

bool ShardSignatureInstance::operator>(ShardSignatureInstance const &other) const {
  return this->tie() > other.tie();
}

bool ShardSignatureInstance::operator<=(ShardSignatureInstance const &other) const {
  return this->tie() <= other.tie();
}

bool ShardSignatureInstance::operator>=(ShardSignatureInstance const &other) const {
  return this->tie() >= other.tie();
}

std::set<OperatorAtomicTaskShardBinding> const &
      ShardSignatureInstance::get_shard_bindings() const
{
  return this->bindings;
}

std::tuple<
  std::set<OperatorAtomicTaskShardBinding> const &
> ShardSignatureInstance::tie() const {
  return std::tie(this->bindings);
}

ParallelTensorDimDegrees
  parallel_tensor_space_for_shard_signature_instance_and_slot(
    ShardSignatureInstance const &shard_signature_instance,
    TensorSlotName slot_name)
{
  std::set<ParallelTensorSpaceCoordinate> task_space_coords = 
    transform(shard_signature_instance.get_shard_bindings(),
              [&](OperatorAtomicTaskShardBinding const &b)
                -> ParallelTensorSpaceCoordinate
              {
                return b.tensor_coords.at(slot_name);
              });

  return assert_unwrap(strict_parallel_tensor_dim_degrees_for_coord_set(task_space_coords));
}

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
  shard_signature_instance_get_ptensor_to_ptensor_mapping(
    ShardSignatureInstance const &shard_signature_instance,
    TensorSlotName lhs_slot_name,
    TensorSlotName rhs_slot_name)
{
  return ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping{
    DimDomainBiuniqueMapping<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>{
      /*coord_mapping=*/
        bidict_from_unstructured_relation(
          transform(shard_signature_instance.get_shard_bindings(),
                    [&](OperatorAtomicTaskShardBinding const &b)
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
        parallel_tensor_space_for_shard_signature_instance_and_slot(shard_signature_instance, lhs_slot_name)),
      /*r_domain=*/dim_domain_from_parallel_tensor_dim_degrees(
        parallel_tensor_space_for_shard_signature_instance_and_slot(shard_signature_instance, rhs_slot_name)),
    },
  };
}

nlohmann::json format_as(ShardSignatureInstance const &inst) {
  nlohmann::json j = inst;
  return j;
}

std::ostream &operator<<(std::ostream &s, ShardSignatureInstance const &inst) {
  return (s << inst);
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::ShardSignatureInstance>::operator()(::FlexFlow::ShardSignatureInstance const &inst) const {
  return get_std_hash(inst.tie());
};

} // namespace std

namespace nlohmann {

::FlexFlow::ShardSignatureInstance adl_serializer<::FlexFlow::ShardSignatureInstance>::from_json(json const &j)
{
  return ::FlexFlow::ShardSignatureInstance{
    j.at("bindings").template get<::FlexFlow::ShardSignatureInstance>(),
  };
}

void adl_serializer<::FlexFlow::ShardSignatureInstance>::to_json(json &j, ::FlexFlow::ShardSignatureInstance const &t)
{
  j["bindings"] = t.get_shard_bindings();
}


} // namespace nlohmann
