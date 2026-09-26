#include "op-attrs/ops/cast.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/containers/transform.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

TensorShape
    cast_get_output_shape(CastAttrs const &attrs, TensorShape const &input) {

  TensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

ParallelTensorDimDegrees cast_get_output_parallel_dim_degrees(
  CastAttrs const &attrs, ParallelTensorDimDegrees const &input_dim_degrees) {

  return input_dim_degrees;
}

ParallelTensorShape
    cast_get_output_parallel_shape(CastAttrs const &attrs, ParallelTensorShape const &input) {

  TensorShape unpar = cast_get_output_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees output_degrees =
      cast_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, output_degrees);
}

StandardOperatorTaskGroup cast_get_task_group(
    CastAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    cast_get_output_parallel_dim_degrees(attrs, input_degrees);

  return StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        ParallelTensorSpaceCoordinate output_coord = input_coord;

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_corods=*/{
            {
              TensorSlotName::INPUT,
              input_coord,
            },
            {
              TensorSlotName::OUTPUT,
              output_coord
            },
          },
          /*task_coord=*/task_coord_matching_parallel_tensor_space_coordinate(output_coord, output_degrees),
        };
      }),
  };
}

ShardSignatureInstance
    cast_get_shard_signature_instance(
          CastAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    cast_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace cast_get_operator_task_space(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    cast_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping cast_get_operator_to_input_mapping(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    cast_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping cast_get_operator_to_output_mapping(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    cast_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}


} // namespace FlexFlow
