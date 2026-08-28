#include "op-attrs/ops/reduce.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape reduce_get_output_shape(ReduceAttrs const &attrs,
                                    TensorShape const &input_shape)
{
  {
    std::set<ff_dim_t> input_dims_set = get_ff_dim_t_set(input_shape.dims);
    ASSERT(
      is_subseteq_of(attrs.axes.unwrap_as_set(), input_dims_set)
    );
  }

  if (attrs.keepdims) {
    return input_shape;
  } else {
    return tensor_shape_drop_dims(
      input_shape,
      [&](ff_dim_t d) -> bool {
        return contains(attrs.axes, d);
      });
  }
}

ParallelTensorDimDegrees reduce_get_output_parallel_dim_degrees(
          ReduceAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  auto only_shard_dims = [&](std::set<parallel_tensor_dim_idx_t> const &ds)
    -> std::set<ff_dim_t>
  {
    filter(
      ds,
      [&](parallel_tensor_dim_idx_t d) -> bool {
        return d.is_shard_dim();
      });
  };

  std::set<ff_dim_t> all_shard_dims =
    only_shard_dims(get_parallel_tensor_dim_indices(input_degrees));

  ASSERT(
    is_subseteq_of(attrs.axes.unwrap_as_set(), all_shard_dims)
  );

  std::set<ff_dim_t> nontrival_shard_dims =
    only_shard_dims(get_nontrivial_parallel_tensor_dim_indices(input_degrees));

  ASSERT(
    are_disjoint(attrs.axes.unwrap_as_set(), nontrival_shard_dims),
  );

  if (attrs.keepdims) {
    return input_degrees;
  } else {
    return parallel_dim_degrees_drop_shard_dims(
      input_degrees,
      [&](ff_dim_t d) -> bool {
        return contains(attrs.axes, d);
      });
  }
}

ParallelTensorShape reduce_get_output_parallel_shape(ReduceAttrs const &,
                                                     ParallelTensorShape const &)
{
  TensorShape output_shape =
      reduce_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(output_shape, output_degrees);
}

OperatorTaskSpace reduce_get_operator_task_space(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
  reduce_get_input_to_output_mapping(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out = make_empty_eq_projection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>();

  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs, input_degrees);

  project_dims(inp_to_out, sum_dim_idx(), sum_dim_idx());
  project_dims(inp_to_out, discard_copy_dim_idx(), discard_copy_dim_idx());

  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();

  ParallelTensorDimDegrees output_degrees =
      flat_get_output_parallel_dim_degrees(attrs, input_degrees);

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{inp_to_out}, input_degrees, output_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reduce_get_operator_to_input_mapping(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping inp_to_out =
      reduce_get_input_to_output_mapping(attrs, input_degrees);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_inp =
      invert_parallel_tensor_space_biunique_mapping(inp_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      reduce_get_operator_to_output_mapping(attrs, input_degrees);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_inp);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reduce_get_operator_to_output_mapping(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_biunique_mapping(
      reduce_get_operator_task_space(attrs, input_degrees),
      output_degrees);
}


} // namespace FlexFlow
