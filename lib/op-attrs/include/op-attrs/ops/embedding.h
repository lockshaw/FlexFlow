#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_EMBEDDING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_EMBEDDING_H

#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

TensorShape embedding_get_output_shape(EmbeddingAttrs const &, TensorShape const &);
TensorShape embedding_get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

ParallelTensorDimDegrees embedding_get_output_parallel_dim_degrees(
  EmbeddingAttrs const &, ParallelTensorDimDegrees const &);
ParallelTensorDimDegrees embedding_get_weights_parallel_dim_degrees(
  EmbeddingAttrs const &, ParallelTensorDimDegrees const &);

ParallelTensorShape embedding_get_output_parallel_shape(EmbeddingAttrs const &, ParallelTensorShape const &);
ParallelTensorShape embedding_get_weights_parallel_shape(EmbeddingAttrs const &, ParallelTensorShape const &);

OperatorTaskSpace embedding_get_operator_task_space(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_input_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_weights_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_output_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

ShardSignatureInstance
    embedding_get_shard_signature_instance(
          EmbeddingAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees);

StandardOperatorTaskGroup embedding_get_task_group(
    EmbeddingAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees);

/**
 * @brief Chosen to match pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/sparse.py#L180-L182
 */
std::map<TensorSlotName, InitializerAttrs> embedding_get_initializers(
    EmbeddingAttrs const &,
    std::optional<InitializerAttrs> const &initializer_attrs = std::nullopt);

} // namespace FlexFlow

#endif
