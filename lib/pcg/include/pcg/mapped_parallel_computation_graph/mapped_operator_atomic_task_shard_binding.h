#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MAPPED_PARALLEL_COMPUTATION_GRAPH_MAPPED_OPERATOR_ATOMIC_TASK_SHARD_BINDING_H

#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.dtg.h"
#include <set>
#include <optional>
#include "op-attrs/tensor_slot_name.dtg.h"
#include "utils/int_ge_two/int_ge_two.h"

namespace FlexFlow {

std::pair<ParallelTensorSpaceCoordinate, MappedOperatorAtomicTaskShardBinding>
  mapped_op_task_shard_binding_project_out_key(MappedOperatorAtomicTaskShardBinding const &, TensorSlotName);

std::optional<nonnegative_int>
  mapped_op_task_shard_bindings_are_k_unique_on_slot(
    std::set<MappedOperatorAtomicTaskShardBinding> const &,
    TensorSlotName);

std::optional<int_ge_two>
  mapped_op_task_shard_bindings_are_strictly_k_unique_on_slot(
    std::set<MappedOperatorAtomicTaskShardBinding> const &,
    TensorSlotName);

bool
  mapped_op_task_shard_bindings_are_unique_on_slot(
    std::set<MappedOperatorAtomicTaskShardBinding> const &,
    TensorSlotName);

} // namespace FlexFlow

#endif
