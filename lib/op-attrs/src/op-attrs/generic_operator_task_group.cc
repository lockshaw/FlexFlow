#include "op-attrs/generic_operator_task_group.h"

namespace FlexFlow {

OperatorAtomicTaskShardBinding
  generic_op_task_group_get_binding_for_task_space_coord(GenericOperatorTaskGroup const &task_group,
                                                         TaskSpaceCoordinate const &task_coord)
{
  return task_group.visit<OperatorAtomicTaskShardBinding>(overload {
    [&](StandardOperatorTaskGroup const &standard_op_task_group)
      -> OperatorAtomicTaskShardBinding
    {
      return standard_op_task_group_get_binding_for_task_space_coord(
        standard_op_task_group,
        task_coord);
    },
    [&](ParallelismOperatorTaskGroup const &parallelism_op_task_group)
      -> OperatorAtomicTaskShardBinding
    {
      return parallelism_op_task_group_get_binding_for_task_space_coord(
        parallelism_op_task_group,
        task_coord);
    }
  });
}

} // namespace FlexFlow
