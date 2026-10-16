#include "op-attrs/ops/reduce.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape reduce_get_output_shape(ReduceAttrs const &attrs,
                                    TensorShape const &input_shape) {
  {
    std::set<ff_dim_t> input_dims_set = get_ff_dim_t_set(input_shape.dims);
    ASSERT(is_subseteq_of(attrs.axes.unwrap_as_set(), input_dims_set));
  }

  if (attrs.keepdims) {
    return input_shape;
  } else {
    return tensor_shape_drop_dims(input_shape, [&](ff_dim_t d) -> bool {
      return contains(attrs.axes, d);
    });
  }
}

ParallelTensorDimDegrees reduce_get_output_parallel_dim_degrees(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  auto only_shard_dims =
      [&](std::set<parallel_tensor_dim_idx_t> const &ds) -> std::set<ff_dim_t> {
    return filtrans(
        ds, [&](parallel_tensor_dim_idx_t const &d) -> std::optional<ff_dim_t> {
          return d.try_require_shard_dim();
        });
  };

  std::set<ff_dim_t> all_shard_dims =
      only_shard_dims(get_parallel_tensor_dim_indices(input_degrees));

  ASSERT(is_subseteq_of(attrs.axes.unwrap_as_set(), all_shard_dims));

  std::set<ff_dim_t> nontrival_shard_dims = only_shard_dims(
      get_nontrivial_parallel_tensor_dim_indices(input_degrees));

  ASSERT(are_disjoint(attrs.axes.unwrap_as_set(), nontrival_shard_dims), );

  if (attrs.keepdims) {
    return input_degrees;
  } else {
    return parallel_dim_degrees_drop_shard_dims(
        input_degrees,
        [&](ff_dim_t d) -> bool { return contains(attrs.axes, d); });
  }
}

ParallelTensorShape
    reduce_get_output_parallel_shape(ReduceAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  TensorShape output_shape =
      reduce_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs,
                                             get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}
NOT_IMPLEMENTED();
}

} // namespace FlexFlow
