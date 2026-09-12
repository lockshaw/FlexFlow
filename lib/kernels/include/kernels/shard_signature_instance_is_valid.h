#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SHARD_SIGNATURE_INSTANCE_IS_VALID_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SHARD_SIGNATURE_INSTANCE_IS_VALID_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "kernels/emulated_parallel_tensor.dtg.h"

namespace FlexFlow {

bool
  shard_signature_instance_is_valid(
    ComputationGraphOpAttrs const &,
    std::map<TensorSlotName, ParallelTensorShape> const &input_shapes,
    std::function<
      std::map<TensorSlotName, GenericTensorAccessorR>(
        std::map<TensorSlotName, GenericTensorAccessorR> const &)> const &run_op,
    int seed);

} // namespace FlexFlow

#endif
