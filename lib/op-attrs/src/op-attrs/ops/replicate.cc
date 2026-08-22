#include "op-attrs/ops/replicate.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/containers/set_union.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"

namespace FlexFlow {

ParallelTensorShape replicate_get_output_parallel_shape(ReplicateAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.replica_dims.discard_copy_degree.value *=
      attrs.replicate_degree;
  return output_shape;
}

ParallelTensorDimDegrees replicate_get_output_parallel_dim_degrees(
    ReplicateAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.discard_copy_degree.value *= attrs.replicate_degree;
  return output_degrees;
}

OperatorTaskSpace replicate_get_operator_task_space(
    ReplicateAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      replicate_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    replicate_get_operator_to_input_mapping(
        ReplicateAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  OperatorTaskSpace op_task_space =
      replicate_get_operator_task_space(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> input_dim_idxs_for_projection =
    set_union(
      get_nontrivial_parallel_tensor_dim_indices(input_degrees),
      std::set{discard_copy_dim_idx()});

  DimProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>
      dim_projection = get_projection_for_op_to_ptensor_identity_mapping(
          operator_task_space_get_dim_idxs(op_task_space), 
          input_dim_idxs_for_projection);

  return operator_ptensor_space_mapping_by_scaling_projection(
      dim_projection, op_task_space, input_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    replicate_get_operator_to_output_mapping(
        ReplicateAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      replicate_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_mapping(
      replicate_get_operator_task_space(attrs, input_degrees), output_degrees);
}

} // namespace FlexFlow
