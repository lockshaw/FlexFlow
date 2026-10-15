#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/standard_operator_task_group.h"
#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"
#include "utils/hash/set.h"
#include "utils/hash/tuple.h"
#include "utils/optional.h"

namespace FlexFlow {

ShardSignatureInstance::ShardSignatureInstance(
    std::set<OperatorAtomicTaskShardBinding> const &shard_bindings)
    : bindings(shard_bindings) {
  // TODO(@lockshaw)(#pr):
}

bool ShardSignatureInstance::operator==(
    ShardSignatureInstance const &other) const {
  return this->tie() == other.tie();
}

bool ShardSignatureInstance::operator!=(
    ShardSignatureInstance const &other) const {
  return this->tie() != other.tie();
}

bool ShardSignatureInstance::operator<(
    ShardSignatureInstance const &other) const {
  return this->tie() < other.tie();
}

bool ShardSignatureInstance::operator>(
    ShardSignatureInstance const &other) const {
  return this->tie() > other.tie();
}

bool ShardSignatureInstance::operator<=(
    ShardSignatureInstance const &other) const {
  return this->tie() <= other.tie();
}

bool ShardSignatureInstance::operator>=(
    ShardSignatureInstance const &other) const {
  return this->tie() >= other.tie();
}

std::set<OperatorAtomicTaskShardBinding> const &
    ShardSignatureInstance::get_shard_bindings() const {
  return this->bindings;
}

std::tuple<std::set<OperatorAtomicTaskShardBinding> const &>
    ShardSignatureInstance::tie() const {
  return std::tie(this->bindings);
}

ParallelTensorDimDegrees
    parallel_tensor_space_for_shard_signature_instance_and_slot(
        ShardSignatureInstance const &shard_signature_instance,
        TensorSlotName slot_name) {
  std::set<ParallelTensorSpaceCoordinate> task_space_coords =
      transform(shard_signature_instance.get_shard_bindings(),
                [&](OperatorAtomicTaskShardBinding const &b)
                    -> ParallelTensorSpaceCoordinate {
                  return b.tensor_coords.at(slot_name);
                });

  return assert_unwrap(
      strict_parallel_tensor_dim_degrees_for_coord_set(task_space_coords));
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

size_t hash<::FlexFlow::ShardSignatureInstance>::operator()(
    ::FlexFlow::ShardSignatureInstance const &inst) const {
  return get_std_hash(inst.tie());
};

} // namespace std

namespace nlohmann {

::FlexFlow::ShardSignatureInstance
    adl_serializer<::FlexFlow::ShardSignatureInstance>::from_json(
        json const &j) {
  return ::FlexFlow::ShardSignatureInstance{
      j.at("bindings").template get<::FlexFlow::ShardSignatureInstance>(),
  };
}

void adl_serializer<::FlexFlow::ShardSignatureInstance>::to_json(
    json &j, ::FlexFlow::ShardSignatureInstance const &t) {
  j["bindings"] = t.get_shard_bindings();
}

} // namespace nlohmann
