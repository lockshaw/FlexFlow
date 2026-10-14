#include "op-attrs/get_standard_operator_task_group.h"
#include "utils/overload.h"

namespace FlexFlow {

StandardOperatorTaskGroup get_standard_operator_task_group(
    ComputationGraphOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const
        &input_dim_degree_binding) {
  return op_attrs.visit<StandardOperatorTaskGroup>(
      overload{[&](auto const &) -> StandardOperatorTaskGroup { PANIC(); }});
}

} // namespace FlexFlow
