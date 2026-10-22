#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ff_ordered/ff_ordered_get_idxs.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "op-attrs/ff_ordered/ff_ordered_restrict_dims_strict.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/all_of.h"
#include "utils/containers/any_of.h"
#include "utils/containers/contains.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter.h"
#include "utils/containers/set_union.h"
#include "utils/containers/vector_of.h"
#include "utils/expected.h"
#include "utils/fmt/set.h"
#include "utils/not_implemented.h"
#include "utils/orthotope/dim_domain.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
  };

  if (attrs.elementwise_affine) {
    result[TensorSlotName::GAMMA] = IncomingTensorRole::WEIGHT;
    result[TensorSlotName::BETA] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

static void check_input_shape(LayerNormAttrs const &attrs,
                              TensorShape const &input_shape) {

  ASSERT(
      all_of(attrs.axes,
             [&](ff_dim_t axis) -> bool {
               return axis.value < get_num_dims(input_shape.dims);
             }),
      fmt::format("LayerNorm axes {} out-of-bounds for input tensor shape {}",
                  attrs.axes,
                  input_shape));
}

TensorShape layer_norm_get_output_shape(LayerNormAttrs const &attrs,
                                        TensorShape const &input_shape) {

  check_input_shape(attrs, input_shape);

  return input_shape;
}

TensorShape layer_norm_get_gamma_weights_shape(LayerNormAttrs const &attrs,
                                               TensorShape const &input_shape) {
  check_input_shape(attrs, input_shape);

  ASSERT(attrs.elementwise_affine,
         "No gamma weights exist for attrs.elementwise_affine = false");

  std::vector<ff_dim_t> normalization_dim_idxs = filter(
      vector_of(ff_ordered_get_idxs(input_shape.dims.ff_ordered)),
      [&](ff_dim_t const &dim_idx) { return contains(attrs.axes, dim_idx); });

  std::vector<positive_int> raw_weight_dims =
      transform(normalization_dim_idxs, [&](ff_dim_t const &dim_idx) {
        return dim_at_idx(input_shape.dims,
                          relative_ff_dim_t_from_ff_dim_t(dim_idx));
      });

  return TensorShape{
      TensorDims{ff_ordered_of(raw_weight_dims)},
      DataType::FLOAT,
  };
}

TensorShape layer_norm_get_beta_weights_shape(LayerNormAttrs const &attrs,
                                              TensorShape const &input_shape) {

  ASSERT(attrs.elementwise_affine,
         "No beta weights exist for attrs.elementwise_affine = false");

  return layer_norm_get_gamma_weights_shape(attrs, input_shape);
}

std::map<TensorSlotName, TensorShape>
    layer_norm_get_weight_shapes(LayerNormAttrs const &attrs,
                                 TensorShape const &input_shape) {

  TensorShape gamma_shape =
      layer_norm_get_gamma_weights_shape(attrs, input_shape);
  TensorShape beta_shape =
      layer_norm_get_beta_weights_shape(attrs, input_shape);

  return std::map<TensorSlotName, TensorShape>{
      {
          TensorSlotName::GAMMA,
          gamma_shape,
      },
      {
          TensorSlotName::BETA,
          beta_shape,
      },
  };
}

static void check_input_parallel_dim_degrees(
    LayerNormAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ASSERT(input_degrees.sum_degree.value == 1,
         fmt::format("Expected sum degree 1, but receieved sum degree {}",
                     input_degrees.sum_degree.value));

  ASSERT(all_of(attrs.axes,
                [&](ff_dim_t axis) -> bool {
                  return get_degree_for_parallel_tensor_dim_idx(
                             input_degrees, shard_dim_idx(axis)) == 1;
                }),
         fmt::format("Expected parallel degree of all dimensions in "
                     "LayerNorm axes {} to be 1, but received input shape {}",
                     attrs.axes,
                     input_degrees));
}

ParallelTensorDimDegrees layer_norm_get_output_parallel_dim_degrees(
    LayerNormAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  check_input_parallel_dim_degrees(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> map_to_sum_dim = {sum_dim_idx(),
                                                        discard_copy_dim_idx()};

  ParallelTensorDimDegrees result = ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{
          dim_domain_get_volume(
              dim_domain_for_ptensor_dim_idxs(input_degrees, map_to_sum_dim)),
      },
      /*discard_copy_degree=*/DiscardCopyDegree{1_p},
      /*shard_degrees=*/input_degrees.shard_degrees,
  };

  ASSERT(get_total_degree_of_ptensor_dim_degrees(input_degrees) ==
         get_total_degree_of_ptensor_dim_degrees(result));

  return result;
}

ParallelTensorDimDegrees layer_norm_get_gamma_weights_parallel_dim_degrees(
    LayerNormAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {

  ASSERT(attrs.elementwise_affine,
         "No gamma weights exist for attrs.elementwise_affine = false");

  check_input_parallel_dim_degrees(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> normalization_dim_idxs =
      transform(attrs.axes, [&](ff_dim_t d) -> parallel_tensor_dim_idx_t {
        return shard_dim_idx(d);
      });

  std::set<parallel_tensor_dim_idx_t> map_to_sum_dim = {discard_copy_dim_idx()};

  std::set<parallel_tensor_dim_idx_t> map_to_discard_copy_dim =
      set_minus(get_parallel_tensor_dim_indices(input_degrees),
                set_union(normalization_dim_idxs, map_to_sum_dim));

  ParallelTensorDimDegrees result = ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{
          dim_domain_get_volume(
              dim_domain_for_ptensor_dim_idxs(input_degrees, map_to_sum_dim)),
      },
      /*discard_copy_degree=*/
      DiscardCopyDegree{
          dim_domain_get_volume(dim_domain_for_ptensor_dim_idxs(
              input_degrees, map_to_discard_copy_dim)),
      },
      /*shard_degrees=*/
      ff_ordered_restrict_dims_strict(input_degrees.shard_degrees, attrs.axes),
  };

  ASSERT(get_total_degree_of_ptensor_dim_degrees(input_degrees) ==
         get_total_degree_of_ptensor_dim_degrees(result));

  return result;
}

ParallelTensorDimDegrees layer_norm_get_beta_weights_parallel_dim_degrees(
    LayerNormAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {

  ASSERT(attrs.elementwise_affine,
         "No beta weights exist for attrs.elementwise_affine = false");

  check_input_parallel_dim_degrees(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> axis_dim_idxs =
      transform(attrs.axes, [&](ff_dim_t d) -> parallel_tensor_dim_idx_t {
        return shard_dim_idx(d);
      });

  std::set<parallel_tensor_dim_idx_t> map_to_sum_dim = {sum_dim_idx(),
                                                        discard_copy_dim_idx()};

  std::set<parallel_tensor_dim_idx_t> map_to_discard_copy_dim =
      set_minus(get_parallel_tensor_dim_indices(input_degrees),
                set_union(axis_dim_idxs, map_to_sum_dim));

  ParallelTensorDimDegrees result = ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{
          dim_domain_get_volume(
              dim_domain_for_ptensor_dim_idxs(input_degrees, map_to_sum_dim)),
      },
      /*discard_copy_degree=*/
      DiscardCopyDegree{
          dim_domain_get_volume(dim_domain_for_ptensor_dim_idxs(
              input_degrees, map_to_discard_copy_dim)),
      },
      /*shard_degrees=*/
      ff_ordered_restrict_dims_strict(input_degrees.shard_degrees, attrs.axes),
  };

  ASSERT(get_total_degree_of_ptensor_dim_degrees(input_degrees) ==
         get_total_degree_of_ptensor_dim_degrees(result));

  return result;
}

std::map<TensorSlotName, ParallelTensorDimDegrees>
    layer_norm_get_weight_parallel_dim_degrees(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_shape) {
  if (attrs.elementwise_affine) {
    return {
        {
            TensorSlotName::GAMMA,
            layer_norm_get_gamma_weights_parallel_dim_degrees(attrs,
                                                              input_shape),
        },
        {
            TensorSlotName::BETA,
            layer_norm_get_beta_weights_parallel_dim_degrees(attrs,
                                                             input_shape),
        },
    };
  } else {
    return {};
  }
}

ParallelTensorShape layer_norm_get_output_parallel_shape(
    LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  TensorShape output_shape =
      layer_norm_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      layer_norm_get_output_parallel_dim_degrees(
          attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

ParallelTensorShape layer_norm_get_gamma_weights_parallel_shape(
    LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  TensorShape gamma_shape =
      layer_norm_get_gamma_weights_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees gamma_degrees =
      layer_norm_get_gamma_weights_parallel_dim_degrees(
          attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(gamma_shape, gamma_degrees);
}

ParallelTensorShape layer_norm_get_beta_weights_parallel_shape(
    LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  TensorShape beta_shape =
      layer_norm_get_beta_weights_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees beta_degrees =
      layer_norm_get_beta_weights_parallel_dim_degrees(
          attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(beta_shape, beta_degrees);
}

std::map<TensorSlotName, ParallelTensorShape>
    layer_norm_get_weight_parallel_shapes(
        LayerNormAttrs const &attrs, ParallelTensorShape const &input_shape) {

  ParallelTensorShape gamma_shape =
      layer_norm_get_gamma_weights_parallel_shape(attrs, input_shape);
  ParallelTensorShape beta_shape =
      layer_norm_get_beta_weights_parallel_shape(attrs, input_shape);

  return std::map<TensorSlotName, ParallelTensorShape>{
      {
          TensorSlotName::GAMMA,
          gamma_shape,
      },
      {
          TensorSlotName::BETA,
          beta_shape,
      },
  };
}

OperatorTaskSpace layer_norm_get_operator_task_space(
    LayerNormAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    layer_norm_get_operator_to_input_mapping(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    layer_norm_get_operator_to_gamma_weights_mapping(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    layer_norm_get_operator_to_beta_weights_mapping(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    layer_norm_get_operator_to_output_mapping(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

std::map<TensorSlotName, InitializerAttrs>
    layer_norm_get_initializers(LayerNormAttrs const &attrs) {
  if (attrs.elementwise_affine) {
    InitializerAttrs gamma_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{1}}}};

    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};

    return {
        {TensorSlotName::GAMMA, gamma_initializer},
        {TensorSlotName::BETA, beta_initializer},
    };
  } else {
    return {};
  }
}

} // namespace FlexFlow
