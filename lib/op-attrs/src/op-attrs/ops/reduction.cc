#include "op-attrs/ops/reduction.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ReductionAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  if (get_sum_degree(input_shape) % attrs.reduction_degree != 0) {
    return tl::unexpected(
        fmt::format("Reduction received tensor with sum degree {}, which is "
                    "not divisible by reduction degree {}",
                    get_sum_degree(input_shape),
                    attrs.reduction_degree));
  }

  ParallelTensorShape output_shape = input_shape;

  output_shape.dims.replica_dims.sum_degree.value = positive_int{
      output_shape.dims.replica_dims.sum_degree.value / attrs.reduction_degree};
  return output_shape;
}

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    ReductionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  positive_int input_degree = input_degrees.sum_degree.value;
  ASSERT(input_degree % attrs.reduction_degree == 0);

  positive_int output_degree = positive_int{
      input_degree / attrs.reduction_degree,
  };

  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.sum_degree = SumDegree{output_degree};

  return output_degrees;
}

OperatorTaskSpace
    get_operator_task_space(ReductionAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees) {
  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      input_degrees);
}

} // namespace FlexFlow
