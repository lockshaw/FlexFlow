#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_conv2d_incoming_tensor_roles(Conv2DAttrs const &);

TensorShape conv2d_get_kernel_shape(Conv2DAttrs const &attrs,
                                    TensorShape const &input);
TensorShape conv2d_get_bias_shape(Conv2DAttrs const &attrs,
                                  TensorShape const &input);
TensorShape conv2d_get_output_shape(Conv2DAttrs const &attrs,
                                    TensorShape const &input);

std::map<TensorSlotName, TensorShape>
    conv2d_get_weight_shapes(Conv2DAttrs const &attrs,
                             TensorShape const &input_shape);

ParallelTensorDimDegrees conv2d_get_kernel_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees);
ParallelTensorDimDegrees conv2d_get_bias_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees);
ParallelTensorDimDegrees conv2d_get_output_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees);

std::map<TensorSlotName, ParallelTensorDimDegrees>
    conv2d_get_weight_parallel_dim_degrees(Conv2DAttrs const &attrs,
                             ParallelTensorDimDegrees const &input_degrees);

ParallelTensorShape
    conv2d_get_kernel_parallel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);
ParallelTensorShape
    conv2d_get_bias_parallel_shape(Conv2DAttrs const &attrs,
                                   ParallelTensorShape const &input_shape);
ParallelTensorShape
    conv2d_get_output_parallel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

std::map<TensorSlotName, ParallelTensorShape>
    conv2d_get_weight_parallel_shapes(Conv2DAttrs const &attrs,
                                      ParallelTensorShape const &input_shape);

OperatorTaskSpace conv2d_get_operator_task_space(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping 
  conv2d_get_operator_to_input_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping 
  conv2d_get_operator_to_kernel_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping 
  conv2d_get_operator_to_bias_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping 
  conv2d_get_operator_to_output_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input);

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
  conv2d_get_operator_to_parallel_tensor_mappings(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input);

std::map<TensorSlotName, InitializerAttrs> conv2d_get_initializers(
    Conv2DAttrs const &attrs,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> kernel_initializer = std::nullopt,
    std::optional<InitializerAttrs> bias_initializer = std::nullopt);

} // namespace FlexFlow

#endif
