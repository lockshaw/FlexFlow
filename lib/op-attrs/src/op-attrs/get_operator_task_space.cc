#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

template <typename T>
static std::vector<T>
    require_only_slots_sequence(std::map<TensorSlotName, T> const &v,
                                std::vector<TensorSlotName> const &slots) {
  nonnegative_int v_num_slots = num_elements(v);
  ASSERT(v_num_slots <= slots.size());

  std::vector<TensorSlotName> expected_slots =
      slice(slots, 0, v_num_slots.unwrap_nonnegative());

  ASSERT(set_of(expected_slots) == keys(v));

  return transform(expected_slots, [&](TensorSlotName const &slot_name) {
    return v.at(slot_name);
  });
};

OperatorTaskSpace get_operator_task_space(
    PCGOperatorAttrs const &attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &inputs_degrees) {
  return task_space_for_standard_operator_task_group(
      get_standard_operator_task_group(attrs, inputs_degrees));
}

} // namespace FlexFlow
