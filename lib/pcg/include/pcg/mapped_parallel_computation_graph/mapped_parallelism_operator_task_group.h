#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLELISM_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_PARALLELISM_OPERATOR_TASK_GROUP_H

#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/parallelism_operator_atomic_task_shard_binding.dtg.h"
#include <nlohmann/json.hpp>
#include "utils/bidict/bidict.h"

namespace FlexFlow {

struct MappedParallelismOperatorTaskGroup {
  MappedParallelismOperatorTaskGroup() = delete;

  explicit MappedParallelismOperatorTaskGroup(
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> const
          &shard_bindings);

  [[nodiscard]] bool operator==(MappedParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator!=(MappedParallelismOperatorTaskGroup const &) const;

  [[nodiscard]] bool operator<(MappedParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>(MappedParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator<=(MappedParallelismOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>=(MappedParallelismOperatorTaskGroup const &) const;

  [[nodiscard]] bidict<MachineSpaceCoordinate,
                       ParallelismOperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> shard_bindings;

private:
  [[nodiscard]] std::tuple<decltype(shard_bindings) const &> tie() const;

  friend struct ::std::hash<MappedParallelismOperatorTaskGroup>;
};

nlohmann::json format_as(::FlexFlow::MappedParallelismOperatorTaskGroup const &);
std::ostream &operator<<(std::ostream &,
                         ::FlexFlow::MappedParallelismOperatorTaskGroup const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::MappedParallelismOperatorTaskGroup> {
  size_t operator()(::FlexFlow::MappedParallelismOperatorTaskGroup const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::MappedParallelismOperatorTaskGroup> {
  static ::FlexFlow::MappedParallelismOperatorTaskGroup from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::MappedParallelismOperatorTaskGroup const &t);
};

} // namespace nlohmann

#endif
