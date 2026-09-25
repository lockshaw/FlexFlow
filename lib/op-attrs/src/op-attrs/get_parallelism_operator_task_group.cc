#include "op-attrs/get_parallelism_operator_task_group.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/combine.h"
#include "utils/containers/require_only_key.h"

namespace FlexFlow {

ParallelismOperatorTaskGroup get_parallelism_operator_task_group(
    ParallelOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &input_dim_degree_binding)
{
  return op_attrs.visit<ParallelismOperatorTaskGroup>(overload {
    [&](CombineAttrs const &attrs) -> ParallelismOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return combine_get_task_group(attrs, input_dim_degrees);
    },
    [&](ReductionAttrs const &attrs) -> ParallelismOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return reduction_get_task_group(attrs, input_dim_degrees);
    },
    [&](RepartitionAttrs const &attrs) -> ParallelismOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return repartition_get_task_group(attrs, input_dim_degrees);
    },
    [&](ReplicateAttrs const &attrs) -> ParallelismOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return replicate_get_task_group(attrs, input_dim_degrees);
    },
  });
}

} // namespace FlexFlow
