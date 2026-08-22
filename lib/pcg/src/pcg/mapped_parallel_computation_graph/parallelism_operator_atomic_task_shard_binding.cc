#include "pcg/mapped_parallel_computation_graph/parallelism_operator_atomic_task_shard_binding.h"

namespace FlexFlow {

OperatorAtomicTaskShardBinding
  operator_atomic_task_shard_binding_from_parallelism_op_binding(
    ParallelismOperatorAtomicTaskShardBinding const &b)
{
  return OperatorAtomicTaskShardBinding{
    /*tensor_coords=*/std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
      {TensorSlotName::INPUT, b.input_coord},
      {TensorSlotName::OUTPUT, b.output_coord},
    },
  };
}

} // namespace FlexFlow
