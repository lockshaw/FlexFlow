#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLELISM_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLELISM_OPERATOR_TASK_GROUP_H

#include "op-attrs/tensor_slot_name.dtg.h"
#include <nlohmann/json.hpp>
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "op-attrs/abstracted_operator_atomic_task_shard_binding.dtg.h"
#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

struct ParallelismOperatorTaskGroup {
  ParallelismOperatorTaskGroup() = delete;

  explicit ParallelismOperatorTaskGroup(
      std::set<AbstractedOperatorAtomicTaskShardBinding> const
          &shard_bindings);

  [[nodiscard]] bool operator==(ParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator!=(ParallelismOperatorTaskGroup const &) const;

  [[nodiscard]] bool operator<(ParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>(ParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator<=(ParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>=(ParallelismOperatorTaskGroup const &) const;

  [[nodiscard]] std::set<AbstractedOperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  std::set<AbstractedOperatorAtomicTaskShardBinding> bindings;

private:
  [[nodiscard]] std::tuple<decltype(bindings) const &> tie() const;

  friend struct ::std::hash<ParallelismOperatorTaskGroup>;
};

OperatorTaskSpace
  task_space_for_parallelism_operator_task_group(
    ParallelismOperatorTaskGroup const &);

ParallelTensorDimDegrees
  parallel_tensor_space_for_parallelism_operator_task_group_and_slot(
    ParallelismOperatorTaskGroup const &,
    TensorSlotName slot_name);

ShardSignatureInstance
  shard_signature_instance_from_parallelism_operator_task_group(
    ParallelismOperatorTaskGroup const &);

OperatorAtomicTaskShardBinding
  parallelism_op_task_group_get_binding_for_task_space_coord(ParallelismOperatorTaskGroup const &,
                                                             TaskSpaceCoordinate const &);

nlohmann::json format_as(::FlexFlow::ParallelismOperatorTaskGroup const &);
std::ostream &operator<<(std::ostream &,
                         ::FlexFlow::ParallelismOperatorTaskGroup const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::ParallelismOperatorTaskGroup> {
  size_t operator()(::FlexFlow::ParallelismOperatorTaskGroup const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::ParallelismOperatorTaskGroup> {
  static ::FlexFlow::ParallelismOperatorTaskGroup from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::ParallelismOperatorTaskGroup const &t);
};

} // namespace nlohmann

#endif
