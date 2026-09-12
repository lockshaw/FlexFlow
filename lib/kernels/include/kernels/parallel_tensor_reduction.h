#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_PARALLEL_TENSOR_REDUCTION_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_PARALLEL_TENSOR_REDUCTION_H

#include "kernels/accessor.h"
#include "kernels/emulated_parallel_tensor.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "op-attrs/operator_atomic_task_shard_binding.dtg.h"
#include "kernels/allocation.h"
#include "op-attrs/ff_dim_t.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, EmulatedParallelTensor>
  parallelize_tensor_operation(
    std::map<TensorSlotName, EmulatedParallelTensor> const &inputs,
    std::set<OperatorAtomicTaskShardBinding> const &bindings,
    std::function<std::map<TensorSlotName, GenericTensorAccessorR>(std::map<TensorSlotName, GenericTensorAccessorR> const &)> const &operation);

EmulatedParallelTensor
  perform_parallel_tensor_reduction(EmulatedParallelTensor const &,
                                    Allocator &);

EmulatedParallelTensor
  perform_parallel_tensor_discard_copy(EmulatedParallelTensor const &);

EmulatedParallelTensor
  perform_parallel_tensor_combination(EmulatedParallelTensor const &,
                                      ff_dim_t dim_idx,
                                      Allocator &); 

GenericTensorAccessorR
  unparallelize_parallel_tensor(EmulatedParallelTensor const &,
                                Allocator &);

} // namespace FlexFlow

#endif
