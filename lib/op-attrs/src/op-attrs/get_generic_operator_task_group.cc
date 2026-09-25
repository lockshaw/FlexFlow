#include "op-attrs/get_generic_operator_task_group.h"
#include "op-attrs/get_standard_operator_task_group.h"
#include "utils/optional.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/get_parallelism_operator_task_group.h"
#include "op-attrs/parallel_op_attrs.h"

namespace FlexFlow {

GenericOperatorTaskGroup get_generic_operator_task_group(
    PCGOperatorAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &input_dim_degrees)
{
  if (is_parallel_op(op_attrs)) {
    ParallelOpAttrs parallel_op_attrs = 
      assert_unwrap(parallel_op_attrs_from_pcg_op_attrs(op_attrs));

    return GenericOperatorTaskGroup{
      get_parallelism_operator_task_group(parallel_op_attrs, input_dim_degrees),
    };
  } else {
    ComputationGraphOpAttrs cg_op_attrs =
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(op_attrs));

    return GenericOperatorTaskGroup{
      get_standard_operator_task_group(cg_op_attrs, input_dim_degrees),
    };
  }
}

} // namespace FlexFlow
