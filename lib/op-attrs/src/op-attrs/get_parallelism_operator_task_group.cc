#include "op-attrs/get_parallelism_operator_task_group.h"
#include "utils/overload.h"

namespace FlexFlow {

ParallelismOperatorTaskGroup get_parallelism_operator_task_group(
    ParallelOpAttrs const &op_attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  return op_attrs.visit<ParallelismOperatorTaskGroup>(overload{
      [&](auto const &) -> ParallelismOperatorTaskGroup { PANIC(); },
  });
}

} // namespace FlexFlow
