#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_STANDARD_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_STANDARD_OPERATOR_TASK_GROUP_H

#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include <nlohmann/json.hpp>
#include "utils/bidict/bidict.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"

namespace FlexFlow {

struct StandardOperatorTaskGroup {
  StandardOperatorTaskGroup() = delete;

  explicit StandardOperatorTaskGroup(
      std::set<AbstractedOperatorAtomicTaskShardBinding> const
          &shard_bindings);

  [[nodiscard]] bool operator==(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator!=(StandardOperatorTaskGroup const &) const;

  [[nodiscard]] bool operator<(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator<=(StandardOperatorTaskGroup const &) const;
  [[nodiscard]] bool operator>=(StandardOperatorTaskGroup const &) const;

  [[nodiscard]] bidict<TaskSpaceCoordinate,
                       OperatorAtomicTaskShardBinding> const &
      get_shard_bindings() const;

private:
  std::set<AbstractedOperatorAtomicTaskShardBinding> bindings;

private:
  [[nodiscard]] std::tuple<decltype(shard_bindings) const &> tie() const;

  friend struct ::std::hash<StandardOperatorTaskGroup>;
};

nlohmann::json format_as(::FlexFlow::StandardOperatorTaskGroup const &);
std::ostream &operator<<(std::ostream &,
                         ::FlexFlow::StandardOperatorTaskGroup const &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::StandardOperatorTaskGroup> {
  size_t operator()(::FlexFlow::StandardOperatorTaskGroup const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::StandardOperatorTaskGroup> {
  static ::FlexFlow::StandardOperatorTaskGroup from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::StandardOperatorTaskGroup const &t);
};

} // namespace nlohmann

#endif
