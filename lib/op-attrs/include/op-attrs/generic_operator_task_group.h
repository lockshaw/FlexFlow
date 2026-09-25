#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GENERIC_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GENERIC_OPERATOR_TASK_GROUP_H

#include "op-attrs/generic_operator_task_group.dtg.h"

namespace FlexFlow {

OperatorAtomicTaskShardBinding
  generic_op_task_group_get_binding_for_task_space_coord(GenericOperatorTaskGroup const &,
                                                         TaskSpaceCoordinate const &);

} // namespace FlexFlow

#endif
