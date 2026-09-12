#include "op-attrs/get_shard_signature_instance.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/get_standard_operator_task_group.h"

namespace FlexFlow {

ShardSignatureInstance get_shard_signature_instance(
    ComputationGraphOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &input_dim_degrees)
{
  StandardOperatorTaskGroup task_group = get_standard_operator_task_group(op_attrs, input_dim_degrees);

  return shard_signature_instance_from_standard_operator_task_group(task_group);
}

} // namespace FlexFlow
