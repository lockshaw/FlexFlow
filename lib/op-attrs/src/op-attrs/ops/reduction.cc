#include "op-attrs/ops/reduction.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"

namespace FlexFlow {

ParallelTensorShape
    reduction_get_output_parallel_shape(ReductionAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  ASSERT(
    get_sum_degree(input_shape) % attrs.reduction_degree == 0,
    fmt::format("Reduction received tensor with sum degree {}, which is "
                "not divisible by reduction degree {}",
                get_sum_degree(input_shape),
                attrs.reduction_degree)
  );

  ParallelTensorShape output_shape = input_shape;

  output_shape.dims.replica_dims.sum_degree.value = positive_int{
      output_shape.dims.replica_dims.sum_degree.value / attrs.reduction_degree};
  return output_shape;
}

ParallelTensorDimDegrees reduction_get_output_parallel_dim_degrees(
    ReductionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {

  positive_int input_degree = input_degrees.sum_degree.value;
  ASSERT(
    input_degree % attrs.reduction_degree == 0,
    input_degree,
    attrs.reduction_degree
  );

  positive_int output_degree = positive_int{
      input_degree / attrs.reduction_degree,
  };

  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.sum_degree = SumDegree{output_degree};

  return output_degrees;
}

ParallelismOperatorTaskGroup reduction_get_task_group(
    ReductionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    reduction_get_output_parallel_dim_degrees(attrs, input_degrees);

  return ParallelismOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        ParallelTensorSpaceCoordinate output_coord = input_coord;
        output_coord.sum_component /= attrs.reduction_degree;

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
          /*task_coord=*/task_coord_matching_parallel_tensor_space_coordinate(input_coord,
                                                                              input_degrees),
        };
      }),
  };
}

OperatorTaskSpace
    reduction_get_operator_task_space(ReductionAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees) {
  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      input_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    reduction_get_operator_to_input_mapping(
        ReductionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceMapping
    reduction_get_operator_to_output_mapping(
        ReductionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
