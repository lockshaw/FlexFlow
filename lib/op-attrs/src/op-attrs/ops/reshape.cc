#include "op-attrs/ops/reshape.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/containers/zip_with_strict.h"

namespace FlexFlow {

TensorShape reshape_get_output_shape(ReshapeAttrs const &attrs,
                                     TensorShape const &input_shape) {
  ASSERT(attrs.shape.data_type == input_shape.data_type);
  ASSERT(get_num_elements(attrs.shape.dims) ==
         get_num_elements(input_shape.dims));

  return attrs.shape;
}

ParallelTensorDimDegrees reshape_get_output_parallel_dim_degrees(
    ReshapeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  // TODO: this can (and probably should) be weakened in specific cases,
  // such as where the leading dimensions are not being modified
  ASSERT(product(input_dim_degrees.shard_degrees) == 1);

  return input_dim_degrees;
}

ParallelTensorShape
    reshape_get_output_parallel_shape(ReshapeAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      reshape_get_output_shape(attrs, get_reduced_shape(input_shape));
  ParallelTensorDimDegrees degrees = reshape_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

StandardOperatorTaskGroup reshape_get_task_group(
    ReshapeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees = 
    reshape_get_output_parallel_dim_degrees(attrs, input_degrees);

  return StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        // TODO(@lockshaw)(#pr):
        NOT_IMPLEMENTED();

        ParallelTensorSpaceCoordinate output_coord = 
              permute_parallel_tensor_space_coordinate(attrs.permutation, input_coord);

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
            {
              TensorSlotName::INPUT,
              input_coord,
            },
            {
              TensorSlotName::OUTPUT,
              output_coord,
            },
          },
          /*task_coord=*/task_coord_matching_parallel_tensor_space_coordinate(output_coord,
                                                                              output_degrees),
        };
      }),
  };
}

ShardSignatureInstance
    reshape_get_shard_signature_instance(
          ReshapeAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace reshape_get_operator_task_space(
    ReshapeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reshape_get_operator_to_input_mapping(
    ReshapeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorShape::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reshape_get_operator_to_output_mapping(
    ReshapeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorShape::OUTPUT);
}

} // namespace FlexFlow
