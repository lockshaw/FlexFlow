#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/not_implemented.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"

namespace FlexFlow {

TensorShape
    softmax_get_output_shape(SoftmaxAttrs const &attrs,
                     TensorShape const &input_shape) {

  ASSERT(
    attrs.dim.value < get_num_dims(input_shape.dims),
    fmt::format("get_output_shape for Softmax received out-of-bounds "
                "attrs.dim {} for input tensor shape {}",
                attrs.dim,
                input_shape)
  );

  return input_shape;
}

ParallelTensorDimDegrees
    softmax_get_output_parallel_dim_degrees(SoftmaxAttrs const &attrs,
                                            ParallelTensorDimDegrees const &input_degrees)
{
  ASSERT(
    input_degrees.sum_degree.value == 1,
    fmt::format("Expected sum degree 1, but received sum degree {}",
                get_sum_degree(input_shape))
  );

  ASSERT(
    shard_dim_at_idx(input_degrees.shard_degrees.at(attrs.dim)) == 1,
    fmt::format("Expected parallel degree of Softmax dimension {} to be 1, "
                "but received input degrees {}",
                attrs.dim,
                input_degrees)
  );

  return input_degrees;
}

ParallelTensorShape
    softmax_get_output_parallel_shape(SoftmaxAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  TensorShape unpar = softmax_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      softmax_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(output_shape, output_degrees);
}

StandardOperatorTaskGroup softmax_get_task_group(
    UpsampleAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    softmax_get_output_parallel_dim_degrees(attrs, input_degrees);

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
      }),
  };
}

ShardSignatureInstance
    softmax_get_shard_signature_instance(
          UpsampleAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    softmax_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace softmax_get_operator_task_space(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group = 
    softmax_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping softmax_get_operator_to_input_mapping(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group = 
    softmax_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping softmax_get_operator_to_output_mapping(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group = 
    softmax_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

} // namespace FlexFlow
