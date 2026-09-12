#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_MULTIHEAD_ATTENTION_PARALLEL_INPUTS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_MULTIHEAD_ATTENTION_PARALLEL_INPUTS_H

#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"

namespace FlexFlow {

MultiHeadAttentionParallelInputs
    parse_attention_parallel_input_shape(ParallelTensorDimDegrees const &input_q,
                                         ParallelTensorDimDegrees const &input_k,
                                         ParallelTensorDimDegrees const &input_v);

} // namespace FlexFlow

#endif
