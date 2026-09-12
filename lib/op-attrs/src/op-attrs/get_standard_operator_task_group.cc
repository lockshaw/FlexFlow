#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/linear.h"
#include "utils/containers/require_two_keys.h"
#include "utils/containers/require_only_key.h"
#include "op-attrs/ops/batch_matmul.h"
#include "utils/overload.h"

namespace FlexFlow {

StandardOperatorTaskGroup get_standard_operator_task_group(
    ComputationGraphOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &input_dim_degree_binding)
{
  return op_attrs.visit<StandardOperatorTaskGroup>(overload {
    [&](BatchMatmulAttrs const &attrs) -> StandardOperatorTaskGroup {
      auto [lhs_input_dim_degrees, rhs_input_dim_degrees] =
        require_two_keys(input_dim_degree_binding, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

      return batch_matmul_get_task_group(attrs, lhs_input_dim_degrees, rhs_input_dim_degrees);
    },
    [&](LinearAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return linear_get_task_group(attrs, input_dim_degrees);
    },
    [&](TransposeAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return transpose_get_task_group(attrs, input_dim_degrees);
    },
    [&](auto const &) -> StandardOperatorTaskGroup {
      PANIC();
    }
  });
}

} // namespace FlexFlow
