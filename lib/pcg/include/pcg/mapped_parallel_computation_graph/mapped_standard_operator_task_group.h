#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_STANDARD_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_STANDARD_OPERATOR_TASK_GROUP_H

#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include <nlohmann/json.hpp>
#include "utils/bidict/bidict.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"

namespace FlexFlow {

struct MappedStandardOperatorTaskGroup {
  MappedStandardOperatorTaskGroup() = delete;

  explicit MappedStandardOperatorTaskGroup(
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const
          &shard_bindings);

  [[nodiscard]] bool operator==(MappedStandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator!=(MappedStandardOperatorTaskGroup const &) const;

  [[nodiscard]] bool operator<(MappedStandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>(MappedStandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator<=(MappedStandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>=(MappedStandardOperatorTaskGroup const &) const;

  [[nodiscard]] bidict<MachineSpaceCoordinate,
                       OperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> shard_bindings;

private:
  [[nodiscard]] std::tuple<decltype(shard_bindings) const &> tie() const;

  friend struct ::std::hash<MappedStandardOperatorTaskGroup>;
};

nlohmann::json format_as(::FlexFlow::MappedStandardOperatorTaskGroup const &);
std::ostream &operator<<(std::ostream &,
                         ::FlexFlow::MappedStandardOperatorTaskGroup const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::MappedStandardOperatorTaskGroup> {
  size_t operator()(::FlexFlow::MappedStandardOperatorTaskGroup const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::MappedStandardOperatorTaskGroup> {
  static ::FlexFlow::MappedStandardOperatorTaskGroup from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::MappedStandardOperatorTaskGroup const &t);
};

} // namespace nlohmann

#endif
