#include "op-attrs/generic_operator_task_group.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/parallelism_operator_task_group.h"

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

OperatorTaskSpace
  task_space_for_generic_operator_task_group(
    GenericOperatorTaskGroup const &task_group)
{
  return task_group.visit<OperatorTaskSpace>(overload {
    [&](StandardOperatorTaskGroup const &standard_op_task_group)
      -> OperatorTaskSpace
    {
      return task_space_for_standard_operator_task_group(standard_op_task_group);
    },
    [&](ParallelismOperatorTaskGroup const &parallelism_op_task_group)
      -> OperatorTaskSpace
    {
      return task_space_for_parallelism_operator_task_group(parallelism_op_task_group);
    }
  });
}

} // namespace FlexFlow
