#include "op-attrs/ops/combine.h"
#include "op-attrs/ff_dim_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/operator_task_space.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(CombineAttrs const &attrs,
                     ParallelTensorShape const &input) {
  ShardParallelDim input_dim = ({
    std::optional<ShardParallelDim> result = try_get_shard_dim_at_idx(
        input, relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim));
    if (!result.has_value()) {
      return tl::unexpected(fmt::format(
          "Failed to get shard dim at index {} in parallel tensor shape {}",
          attrs.combine_dim,
          input));
    }

    result.value();
  });

  if (input_dim.degree % attrs.combine_degree != 0) {
    return tl::unexpected(
        fmt::format("Combine received tensor containing parallel dim {} with "
                    "degree {}, which is not divisible by combine degree {}",
                    attrs.combine_dim,
                    input_dim.degree,
                    attrs.combine_degree));
  }

  ParallelTensorShape output = input;
  relative_ff_dim_t combine_dim =
      relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim);
  shard_dim_at_idx(output, combine_dim).degree = positive_int{
      shard_dim_at_idx(output, combine_dim).degree / attrs.combine_degree};

  return output;
}

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    CombineAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  positive_int input_degree = input_degrees.shard_degrees.at(attrs.combine_dim);
  ASSERT(input_degree % attrs.combine_degree == 0);

  positive_int output_degree = positive_int{
    input_degree / attrs.combine_degree,
  };

  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.shard_degrees
      .at(relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim))
      = output_degree;

  return output_degrees;

}

OperatorTaskSpace
    get_operator_task_space(CombineAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees)
{
  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      input_degrees);
}

} // namespace FlexFlow
