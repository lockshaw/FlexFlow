#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_SHARD_SIGNATURE_INSTANCE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_SHARD_SIGNATURE_INSTANCE_H

#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"

namespace FlexFlow {

ShardSignatureInstance get_shard_signature_instance(
    ComputationGraphOpAttrs const &,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &input_dim_degrees);

} // namespace FlexFlow

#endif
