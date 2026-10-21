#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_PARALLELISM_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_PARALLELISM_OPERATOR_TASK_GROUP_H

#include "op-attrs/parallel_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallelism_operator_task_group.h"

namespace FlexFlow {

ParallelismOperatorTaskGroup get_parallelism_operator_task_group(
    ParallelOpAttrs const &, ParallelTensorDimDegrees const &input_dim_degrees);

} // namespace FlexFlow

#endif
