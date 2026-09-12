#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ATTENTION_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.dtg.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.dtg.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include <tl/expected.hpp>
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs const &);

TensorShape
    attention_get_weights_shape(MultiHeadAttentionAttrs const &,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v);
TensorShape
    attention_get_input_bias_shape(MultiHeadAttentionAttrs const &,
                         TensorShape const &input_q,
                         TensorShape const &input_k,
                         TensorShape const &input_v);
TensorShape
    attention_get_output_bias_shape(MultiHeadAttentionAttrs const &,
                          TensorShape const &input_q,
                          TensorShape const &input_k,
                          TensorShape const &input_v);
TensorShape
    attention_get_output_shape(MultiHeadAttentionAttrs const &,
                     TensorShape const &input_q,
                     TensorShape const &input_k,
                     TensorShape const &input_v);

std::map<TensorSlotName, TensorShape>
    attention_get_weight_shapes(MultiHeadAttentionAttrs const &,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v);

ParallelTensorDimDegrees
    attention_get_weights_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                      ParallelTensorDimDegrees const &input_q,
                      ParallelTensorDimDegrees const &input_k,
                      ParallelTensorDimDegrees const &input_v);
ParallelTensorDimDegrees
    attention_get_input_bias_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                         ParallelTensorDimDegrees const &input_q,
                         ParallelTensorDimDegrees const &input_k,
                         ParallelTensorDimDegrees const &input_v);
ParallelTensorDimDegrees
    attention_get_output_bias_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                          ParallelTensorDimDegrees const &input_q,
                          ParallelTensorDimDegrees const &input_k,
                          ParallelTensorDimDegrees const &input_v);
ParallelTensorDimDegrees
    attention_get_output_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                     ParallelTensorDimDegrees const &input_q,
                     ParallelTensorDimDegrees const &input_k,
                     ParallelTensorDimDegrees const &input_v);

std::map<TensorSlotName, ParallelTensorDimDegrees>
    attention_get_weight_parallel_dim_degrees(MultiHeadAttentionAttrs const &,
                      ParallelTensorDimDegrees const &input_q,
                      ParallelTensorDimDegrees const &input_k,
                      ParallelTensorDimDegrees const &input_v);

ParallelTensorDims
    attention_get_weights_parallel_dims(MultiHeadAttentionAttrs const &,
                              ParallelTensorShape const &input_q,
                              ParallelTensorShape const &input_k,
                              ParallelTensorShape const &input_v);
ParallelTensorDims
    attention_get_input_bias_parallel_dims(MultiHeadAttentionAttrs const &,
                                 ParallelTensorShape const &input_q,
                                 ParallelTensorShape const &input_k,
                                 ParallelTensorShape const &input_v);
ParallelTensorDims
    attention_get_output_bias_parallel_dims(MultiHeadAttentionAttrs const &,
                                  ParallelTensorShape const &input_q,
                                  ParallelTensorShape const &input_k,
                                  ParallelTensorShape const &input_v);

ParallelTensorShape
    attention_get_weights_parallel_shape(MultiHeadAttentionAttrs const &,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v);
ParallelTensorShape
    attention_get_input_bias_parallel_shape(MultiHeadAttentionAttrs const &,
                         ParallelTensorShape const &input_q,
                         ParallelTensorShape const &input_k,
                         ParallelTensorShape const &input_v);
ParallelTensorShape
    attention_get_output_bias_parallel_shape(MultiHeadAttentionAttrs const &,
                          ParallelTensorShape const &input_q,
                          ParallelTensorShape const &input_k,
                          ParallelTensorShape const &input_v);
ParallelTensorShape
    attention_get_output_parallel_shape(MultiHeadAttentionAttrs const &,
                     ParallelTensorShape const &input_q,
                     ParallelTensorShape const &input_k,
                     ParallelTensorShape const &input_v);

std::map<TensorSlotName, ParallelTensorShape>
    attention_get_weight_parallel_shapes(MultiHeadAttentionAttrs const &,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v);

OperatorTaskSpace attention_get_operator_task_space(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_query_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_key_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_value_input_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_weights_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_input_bias_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_output_bias_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  attention_get_operator_to_output_mapping(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
  attention_get_operator_to_parallel_tensor_mappings(
    MultiHeadAttentionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_q,
    ParallelTensorDimDegrees const &input_k,
    ParallelTensorDimDegrees const &input_v);

std::map<TensorSlotName, InitializerAttrs>
    attention_get_initializers(
        MultiHeadAttentionAttrs const &,
        TensorShape const &input_q,
        TensorShape const &input_k,
        TensorShape const &input_v,
        std::optional<InitializerAttrs> const &weights_initializer =
            std::nullopt,
        std::optional<InitializerAttrs> const &input_bias_initializer =
            std::nullopt,
        std::optional<InitializerAttrs> const &output_bias_initializer =
            std::nullopt);

} // namespace FlexFlow

#endif
