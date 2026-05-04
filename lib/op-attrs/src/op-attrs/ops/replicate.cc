#include "op-attrs/ops/replicate.h"
#include "op-attrs/operator_task_space.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReplicateAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.replica_dims.discard_copy_degree.value *=
      attrs.replicate_degree;
  return output_shape;
}

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    ReplicateAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.discard_copy_degree.value *= attrs.replicate_degree;
  return output_degrees;

}

OperatorTaskSpace
    get_operator_task_space(ReplicateAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees = get_output_parallel_dim_degrees(
      attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceMapping
    get_input_to_output_mapping(ReplicateAttrs const &attrs,
                                ParallelTensorDimDegrees const &input_degrees) {

  num_tensor_dims_t input_num_dims =
      get_ptensor_dim_degrees_num_tensor_dims(input_degrees);

  DownProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out = make_empty_down_projection<parallel_tensor_dim_idx_t,
                                              parallel_tensor_dim_idx_t>();

  ff_dim_t input_channel_dim =
      ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, input_num_dims);

  num_tensor_dims_t output_num_dims = input_num_dims;
  ff_dim_t output_channel_dim =
      ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, output_num_dims);

  project_dims(inp_to_out,
               /*from=*/{},
               /*onto=*/sum_dim_idx());
  project_dims(inp_to_out,
               /*from=*/{},
               /*onto=*/shard_dim_idx(output_channel_dim));

  for (ff_dim_t const &idx : slice(tensor_dims_range(input_num_dims), 0, -1)) {
    project_dims(inp_to_out,
                 /*from=*/{shard_dim_idx(idx)},
                 /*onto=*/shard_dim_idx(idx));
  }

  ParallelTensorDimDegrees output_degrees =
      get_output_parallel_dim_degrees(attrs, input_degrees);

  return parallel_tensor_space_mapping_from_projection(
      DimProjection{inp_to_out}, input_degrees, output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    ReplicateAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
}

} // namespace FlexFlow
