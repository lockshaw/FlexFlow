#include "op-attrs/ops/flat.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/ff_ordered/ff_ordered_slice_inclusive.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/all_of.h"
#include "utils/containers/product.h"
#include <cassert>
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "utils/orthotope/eq_projection.h"
#include "op-attrs/ff_dim_t.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"

namespace FlexFlow {

TensorShape flat_get_output_shape(FlatAttrs const &attrs,
                                  TensorShape const &input_shape) {
  FFOrdered<positive_int> leading_dims = ff_ordered_slice(
      ff_ordered(input_shape.dims), ff_dim_t{0_n}, attrs.start_dim);
  FFOrdered<positive_int> flattened_dims = ff_ordered_slice_inclusive(
      ff_ordered(input_shape.dims), attrs.start_dim, attrs.end_dim);
  FFOrdered<positive_int> trailing_dims =
      ff_ordered_slice(ff_ordered(input_shape.dims),
                       add_to_ff_dim(attrs.end_dim, 1),
                       std::nullopt);

  if (flattened_dims.empty()) {
    return input_shape;
  }

  return TensorShape{
      TensorDims{
          ff_ordered_concat(std::vector{
              leading_dims,
              FFOrdered{product(flattened_dims)},
              trailing_dims,
          }),
      },
      input_shape.data_type,
  };
}

ParallelTensorDimDegrees flat_get_output_parallel_dim_degrees(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  FFOrdered<positive_int> flattened_dim_degrees = ff_ordered_slice_inclusive(
      input_degrees.shard_degrees, attrs.start_dim, attrs.end_dim);

  if (flattened_dim_degrees.empty()) {
    return input_degrees;
  }

  ASSERT(all_of(flattened_dim_degrees,
                [](positive_int degree) { return degree == 1; }),
         "flat_get_output_parallel_dim_degrees expected all shard degrees of "
         "flattened dimensions to be 1",
         attrs,
         input_degrees,
         flattened_dim_degrees);

  return ParallelTensorDimDegrees{
      /*sum_degree=*/input_degrees.sum_degree,
      /*discard_copy_degree=*/input_degrees.discard_copy_degree,
      /*shard_degrees=*/
      ff_ordered_concat(std::vector{
          ff_ordered_slice(
              input_degrees.shard_degrees, ff_dim_t{0_n}, attrs.start_dim),
          FFOrdered{product(flattened_dim_degrees)},
          ff_ordered_slice(input_degrees.shard_degrees,
                           add_to_ff_dim(attrs.end_dim, 1),
                           std::nullopt),
      }),
  };
}

ParallelTensorShape
    flat_get_output_parallel_shape(FlatAttrs const &attrs,
                                   ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      flat_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = flat_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

OperatorTaskSpace flat_get_operator_task_space(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
      flat_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    flat_get_input_to_output_mapping(FlatAttrs const &attrs,
                                ParallelTensorDimDegrees const &input_degrees) {

  auto ff_dim_to_pt_dim = [](ff_dim_t d) -> parallel_tensor_dim_idx_t {
    return parallel_tensor_dim_idx_t{d};
  };

  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out = make_empty_eq_projection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>();

  project_dims(inp_to_out, sum_dim_idx(), sum_dim_idx());
  project_dims(inp_to_out, discard_copy_dim_idx(), discard_copy_dim_idx());

  auto compute_pre_start_output_dim = [&](ff_dim_t input_dim) -> ff_dim_t {
    ASSERT(input_dim < attrs.start_dim);
    return input_dim;
  };

  auto compute_in_flat_output_dim = [&](ff_dim_t input_dim) -> ff_dim_t {
    ASSERT(input_dim >= attrs.start_dim);
    ASSERT(input_dim <= attrs.end_dim);
    return attrs.start_dim;
  };

  auto compute_post_end_output_dim = [&](ff_dim_t input_dim) -> ff_dim_t {
    ASSERT(input_dim > attrs.end_dim);
    int offset = attrs.start_dim.value.int_from_nonnegative_int()
      - attrs.end_dim.value.int_from_nonnegative_int();

    return add_to_ff_dim(input_dim, offset);
  };

  auto compute_output_dim = [&](ff_dim_t input_dim) -> ff_dim_t {
    if (input_dim < attrs.start_dim) {
      return compute_pre_start_output_dim(input_dim);
    } else if (input_dim >= attrs.start_dim && input_dim <= attrs.end_dim) {
      return compute_in_flat_output_dim(input_dim);
    } else {
      return compute_post_end_output_dim(input_dim);
    }
  };

  for (ff_dim_t const &input_dim: tensor_dims_range(get_ptensor_dim_degrees_num_tensor_dims(input_degrees))) {
    ff_dim_t output_dim = compute_output_dim(input_dim);
    project_dims(inp_to_out, shard_dim_idx(input_dim), shard_dim_idx(output_dim));
  }

  ParallelTensorDimDegrees output_degrees =
      flat_get_output_parallel_dim_degrees(attrs, input_degrees);

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{inp_to_out}, input_degrees, output_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping flat_get_operator_to_input_mapping(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping inp_to_out =
      flat_get_input_to_output_mapping(attrs, input_degrees);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_inp =
      invert_parallel_tensor_space_biunique_mapping(inp_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      flat_get_operator_to_output_mapping(attrs, input_degrees);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_inp);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping flat_get_operator_to_output_mapping(
    FlatAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
      flat_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_biunique_mapping(
      flat_get_operator_task_space(attrs, input_degrees),
      output_degrees);
}

} // namespace FlexFlow
