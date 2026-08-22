#include "op-attrs/ops/repartition.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include <libassert/assert.hpp>
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/containers/set_union.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"

namespace FlexFlow {

ParallelTensorShape repartition_get_output_parallel_shape(
    RepartitionAttrs const &attrs, ParallelTensorShape const &input_shape) {
  ASSERT(input_shape.dims.shard_dims.idx_is_valid(attrs.repartition_dim),
         attrs,
         input_shape);

  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.shard_dims
      .at(relative_ff_dim_t_from_ff_dim_t(attrs.repartition_dim))
      .degree *= attrs.repartition_degree;
  return output_shape;
}

ParallelTensorDimDegrees repartition_get_output_parallel_dim_degrees(
    RepartitionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.shard_degrees.at(relative_ff_dim_t_from_ff_dim_t(
      attrs.repartition_dim)) *= attrs.repartition_degree;
  return output_degrees;
}

OperatorTaskSpace repartition_get_operator_task_space(
    RepartitionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      repartition_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping
    repartition_get_operator_to_input_mapping(
        RepartitionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  OperatorTaskSpace op_task_space =
      repartition_get_operator_task_space(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> input_dim_idxs_for_projection =
    set_union(
      get_nontrivial_parallel_tensor_dim_indices(input_degrees),
      std::set{shard_dim_idx(attrs.repartition_dim)});

  DimProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>
      dim_projection = get_projection_for_op_to_ptensor_identity_mapping(
          operator_task_space_get_dim_idxs(op_task_space),
          input_dim_idxs_for_projection);

  return operator_ptensor_space_mapping_by_scaling_projection(
      dim_projection, op_task_space, input_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    repartition_get_operator_to_output_mapping(
        RepartitionAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees output_degrees =
      repartition_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_biunique_mapping(
      repartition_get_operator_task_space(attrs, input_degrees),
      output_degrees);
}

} // namespace FlexFlow
