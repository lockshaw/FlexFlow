#include "op-attrs/ops/reshape.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/containers/zip_with_strict.h"

namespace FlexFlow {

TensorShape reshape_get_output_shape(ReshapeAttrs const &attrs,
                                     TensorShape const &input_shape) {

  TensorDims leading_dims =
      tensor_dims_remove_trailing_dims(input_shape.dims, attrs.core_input_dims);

  TensorDims output_dims =
      concat_tensor_dims(leading_dims, attrs.core_output_dims);

  return TensorShape{
      /*dims=*/output_dims,
      /*data_type=*/input_shape.data_type,
  };
}

/*
// TODO(@lockshaw)(#pr):
static
std::set<parallel_tensor_dim_idx_t> get_dims_allowed_parallel(ReshapeAttrs const &attrs)
{
  num_tensor_dims_t max_dims = std::max(
    get_num_dims(attrs.core_input_dims),
    get_num_dims(attrs.core_output_dims));

  std::set<ff_dim_t>
    leading_dims = set_of(
      take_while(
        ff_dim_range(max_dims.nonnegative_int_from_num_tensor_dims()),
        [&](ff_dim_t d) -> bool {
          positive_int input_dim_size = dim_at_idx(attrs.core_input_dims, d);
          positive_int output_dim_size = dim_at_idx(attrs.core_output_dims, d);

          return input_dim_size == output_dim_size;
        }));

  std::set<parallel_tensor_dim_idx_t> dims_allowed_parallel =
    set_union(
      transform(leading_dims,
                [&](ff_dim_t d) -> parallel_tensor_dim_idx_t {
                  return shard_dim_idx(d);
                }),
      std::set{sum_dim_idx(), discard_copy_dim_idx()});

  return dims_allowed_parallel;
}
*/

ParallelTensorDimDegrees reshape_get_output_parallel_dim_degrees(
    ReshapeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {

  FFOrdered<positive_int> input_trailing_dim_degrees =
      ff_ordered_transform(attrs.core_input_dims.ff_ordered,
                           [&](positive_int) -> positive_int { return 1_p; });

  ParallelTensorDimDegrees leading_dim_degrees =
      parallel_tensor_dim_degrees_remove_trailing_dims(
          input_dim_degrees, input_trailing_dim_degrees);

  FFOrdered<positive_int> output_trailing_dim_degrees =
      ff_ordered_transform(attrs.core_output_dims.ff_ordered,
                           [&](positive_int) -> positive_int { return 1_p; });

  return parallel_tensor_dim_degrees_append_trailing_dims(
      leading_dim_degrees, output_trailing_dim_degrees);
}

ParallelTensorShape
    reshape_get_output_parallel_shape(ReshapeAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      reshape_get_output_shape(attrs, get_reduced_shape(input_shape));
  ParallelTensorDimDegrees degrees = reshape_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

} // namespace FlexFlow
