#include "op-attrs/ops/combine.h"
#include "op-attrs/ff_dim_t.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <libassert/assert.hpp>
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/containers/set_union.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"

namespace FlexFlow {

ParallelTensorShape
    combine_get_output_parallel_shape(CombineAttrs const &attrs,
                                      ParallelTensorShape const &input) {
  ShardParallelDim input_dim = shard_dim_at_idx(
      input, relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim));

  ASSERT(input_dim.degree % attrs.combine_degree == 0,
         fmt::format("Combine received tensor containing parallel dim {} with "
                     "degree {}, which is not divisible by combine degree {}",
                     attrs.combine_dim,
                     input_dim.degree,
                     attrs.combine_degree),
         input);

  ParallelTensorShape output = input;
  relative_ff_dim_t combine_dim =
      relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim);
  shard_dim_at_idx(output, combine_dim).degree = positive_int{
      shard_dim_at_idx(output, combine_dim).degree / attrs.combine_degree};

  return output;
}

ParallelTensorDimDegrees combine_get_output_parallel_dim_degrees(
    CombineAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  positive_int input_degree = input_degrees.shard_degrees.at(attrs.combine_dim);
  ASSERT(input_degree % attrs.combine_degree == 0);

  positive_int output_degree = positive_int{
      input_degree / attrs.combine_degree,
  };

  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.shard_degrees.at(
      relative_ff_dim_t_from_ff_dim_t(attrs.combine_dim)) = output_degree;

  return output_degrees;
}

OperatorTaskSpace
    combine_get_operator_task_space(CombineAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees) {
  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      input_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    combine_get_operator_to_input_mapping(
        CombineAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees)
{
  return get_identity_biunique_mapping(
      combine_get_operator_task_space(attrs, input_degrees),
      input_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    combine_get_operator_to_output_mapping(
        CombineAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees)
{
  OperatorTaskSpace op_task_space =
      combine_get_operator_task_space(attrs, input_degrees);

  ParallelTensorDimDegrees output_degrees =
      combine_get_output_parallel_dim_degrees(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> output_dim_idxs_for_projection =
    set_union(
      get_nontrivial_parallel_tensor_dim_indices(output_degrees),
      std::set{shard_dim_idx(attrs.combine_dim)});

  DimProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>
      dim_projection = get_projection_for_op_to_ptensor_identity_mapping(
          operator_task_space_get_dim_idxs(op_task_space),
          output_dim_idxs_for_projection);

  return operator_ptensor_space_mapping_by_scaling_projection(
      dim_projection, op_task_space, output_degrees);
}


} // namespace FlexFlow
