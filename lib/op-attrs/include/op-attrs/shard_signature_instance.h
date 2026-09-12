#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHARD_SIGNATURE_INSTANCE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHARD_SIGNATURE_INSTANCE_H

#include "op-attrs/tensor_slot_name.dtg.h"
#include <nlohmann/json.hpp>
#include "utils/bidict/bidict.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "op-attrs/abstracted_operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"

namespace FlexFlow {

struct ShardSignatureInstance {
  ShardSignatureInstance() = delete;

  explicit ShardSignatureInstance(
      std::set<OperatorAtomicTaskShardBinding> const
          &shard_bindings);

  [[nodiscard]] bool operator==(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator!=(ShardSignatureInstance const &) const;

  [[nodiscard]] bool operator<(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator>(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator<=(ShardSignatureInstance const &) const;
  [[nodiscard]] bool operator>=(ShardSignatureInstance const &) const;

  [[nodiscard]] std::set<OperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  std::set<OperatorAtomicTaskShardBinding> bindings;

private:
  [[nodiscard]] std::tuple<decltype(bindings) const &> tie() const;

  friend struct ::std::hash<ShardSignatureInstance>;
};

ParallelTensorDimDegrees
  parallel_tensor_space_for_shard_signature_instance_and_slot(
    ShardSignatureInstance const &,
    TensorSlotName slot_name);

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
  shard_signature_instance_get_ptensor_to_ptensor_mapping(
    ShardSignatureInstance const &,
    TensorSlotName lhs_slot_name,
    TensorSlotName rhs_slot_name);

nlohmann::json format_as(ShardSignatureInstance const &);
std::ostream &operator<<(std::ostream &, ShardSignatureInstance const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::ShardSignatureInstance> {
  size_t operator()(::FlexFlow::ShardSignatureInstance const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::ShardSignatureInstance> {
  static ::FlexFlow::ShardSignatureInstance from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::ShardSignatureInstance const &t);
};

} // namespace nlohmann

#endif
