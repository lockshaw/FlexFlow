#include "op-attrs/ops/dropout.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/not_implemented.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/containers/transform.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

TensorShape dropout_get_output_shape(DropoutAttrs const &,
                             TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorDimDegrees
    dropout_get_output_parallel_dim_degrees(DropoutAttrs const &,
                                            ParallelTensorDimDegrees const &input_degrees)
{
  ASSERT(
    input_degrees.sum_degree.value == 1,
    fmt::format("Expected sum degree 1, but receieved sum degree {}",
                input_degrees.sum_degree.value)
  );

  ASSERT(
    input_degrees.discard_copy_degree.value == 1,
    fmt::format(
        "Expected discard copy degree 1, but received discard copy degree {}",
        input_degrees.discard_copy_degree.value)
  );

  return input_degrees;
}

ParallelTensorShape
    dropout_get_output_parallel_shape(DropoutAttrs const &attrs,
                     ParallelTensorShape const &input_shape)
{
  TensorShape unpar = dropout_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      dropout_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, output_degrees);
}

StandardOperatorTaskGroup dropout_get_task_group(
    DropoutAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    dropout_get_output_parallel_dim_degrees(attrs, input_degrees);

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
    dropout_get_shard_signature_instance(
          DropoutAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    dropout_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace dropout_get_operator_task_space(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    dropout_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping dropout_get_operator_to_input_mapping(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    dropout_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping dropout_get_operator_to_output_mapping(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    dropout_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

} // namespace FlexFlow
