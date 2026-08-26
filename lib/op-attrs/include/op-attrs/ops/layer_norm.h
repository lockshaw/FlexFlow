#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LAYER_NORM_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LAYER_NORM_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &);

TensorShape layer_norm_get_output_shape(LayerNormAttrs const &, TensorShape const &);
TensorShape layer_norm_get_gamma_weights_shape(LayerNormAttrs const &, TensorShape const &);
TensorShape layer_norm_get_beta_weights_shape(LayerNormAttrs const &, TensorShape const &);

std::map<TensorSlotName, TensorShape>
    layer_norm_get_weight_shapes(LayerNormAttrs const &attrs,
                      TensorShape const &input_shape);

ParallelTensorDimDegrees
    layer_norm_get_output_parallel_dim_degrees(LayerNormAttrs const &, ParallelTensorDimDegrees const &);
ParallelTensorDimDegrees
    layer_norm_get_gamma_weights_parallel_dim_degrees(LayerNormAttrs const &,
                            ParallelTensorDimDegrees const &);
ParallelTensorDimDegrees
    layer_norm_get_beta_weights_parallel_dim_degrees(LayerNormAttrs const &, ParallelTensorDimDegrees const &);

std::map<TensorSlotName, ParallelTensorDimDegrees>
    layer_norm_get_weight_parallel_dim_degrees(LayerNormAttrs const &attrs,
                      ParallelTensorDimDegrees const &input_shape);

ParallelTensorShape
    layer_norm_get_output_parallel_shape(LayerNormAttrs const &, ParallelTensorShape const &);
ParallelTensorShape
    layer_norm_get_gamma_weights_parallel_shape(LayerNormAttrs const &,
                            ParallelTensorShape const &);
ParallelTensorShape
    layer_norm_get_beta_weights_parallel_shape(LayerNormAttrs const &, ParallelTensorShape const &);

std::map<TensorSlotName, ParallelTensorShape>
    layer_norm_get_weight_parallel_shapes(LayerNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

OperatorTaskSpace layer_norm_get_operator_task_space(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_input_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_gamma_weights_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_beta_weights_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_output_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

/**
 * @brief Chosen to match pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/normalization.py#L210-L214
 */
std::map<TensorSlotName, InitializerAttrs>
    layer_norm_get_initializers(LayerNormAttrs const &attrs);

} // namespace FlexFlow

#endif
