#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/all_of.h"
#include "utils/containers/contains.h"
#include "utils/fmt/set.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_union.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/ff_ordered/ff_ordered_restrict_dims_strict.h"
#include "utils/optional.h"

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

std::set<TensorSlotName> layer_norm_get_slots(LayerNormAttrs const &attrs) {
  std::set<TensorSlotName> result = {
    TensorSlotName::INPUT,
    TensorSlotName::OUTPUT,
  };

  if (attrs.elementwise_affine) {
    result.insert(TensorSlotName::GAMMA);
    result.insert(TensorSlotName::BETA);
  };

  return result;
}

static void
    check_input_shape(LayerNormAttrs const &attrs,
                      TensorShape const &input_shape) {

  ASSERT(
    all_of(attrs.axes,
           [&](ff_dim_t axis) -> bool {
             return axis.value < get_num_dims(input_shape.dims);
           }),
    fmt::format(
        "LayerNorm axes {} out-of-bounds for input tensor shape {}",
        attrs.axes,
        input_shape)
  );
}

TensorShape
    layer_norm_get_output_shape(LayerNormAttrs const &attrs,
                                TensorShape const &input_shape) {
  check_input_shape(attrs, input_shape);

  return input_shape;
}

TensorShape
    layer_norm_get_gamma_weights_shape(LayerNormAttrs const &attrs,
                                       TensorShape const &input_shape) {
  check_input_shape(attrs, input_shape);

  ASSERT(attrs.elementwise_affine, "No gamma weights exist for attrs.elementwise_affine = false");

  return TensorShape{
    tensor_dims_filtrans_with_idx(
      input_shape.dims,
      [&](ff_dim_t const &dim_idx, positive_int dim_size) -> std::optional<positive_int> {
        if (contains(attrs.axes, dim_idx)) {
          return dim_size;
        } else {
          return std::nullopt;
        }
      }
    ),
    DataType::FLOAT,
  };
}

TensorShape
    layer_norm_get_beta_weights_shape(LayerNormAttrs const &attrs,
                                      TensorShape const &input_shape) {

  ASSERT(attrs.elementwise_affine, "No beta weights exist for attrs.elementwise_affine = false");

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

static void
    check_input_parallel_dim_degrees(LayerNormAttrs const &attrs,
                                     ParallelTensorDimDegrees const &input_degrees) {
  ASSERT(
    input_degrees.sum_degree.value == 1,
    fmt::format("Expected sum degree 1, but receieved sum degree {}",
                input_degrees.sum_degree.value)
  );

  ASSERT(
    all_of(attrs.axes,
           [&](ff_dim_t axis) -> bool {
             return get_degree_for_parallel_tensor_dim_idx(input_degrees, shard_dim_idx(axis)) == 1;
           }),
    fmt::format("Expected parallel degree of all dimensions in "
                "LayerNorm axes {} to be 1, but received input shape {}",
                attrs.axes,
                input_degrees)
  );
}

ParallelTensorDimDegrees
    layer_norm_get_output_parallel_dim_degrees(LayerNormAttrs const &attrs,
                                               ParallelTensorDimDegrees const &input_degrees)
{
  check_input_parallel_dim_degrees(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> map_to_sum_dim = {sum_dim_idx(), discard_copy_dim_idx()};
  
  ParallelTensorDimDegrees result = ParallelTensorDimDegrees{
    /*sum_degree=*/SumDegree{
      dim_domain_get_volume(
        dim_domain_for_ptensor_dims(
          input_degrees,
          map_to_sum_dim)),
    },
    /*discard_copy_degree=*/DiscardCopyDegree{1_p},
    /*shard_degrees=*/input_degrees.shard_degrees,
  };

  ASSERT(
    get_total_degree_of_ptensor_dim_degrees(input_degrees) 
    == 
    get_total_degree_of_ptensor_dim_degrees(result)
  );

  return result;
}


ParallelTensorDimDegrees
    layer_norm_get_gamma_weights_parallel_dim_degrees(LayerNormAttrs const &attrs,
                                                      ParallelTensorDimDegrees const &input_degrees)
{
  ASSERT(attrs.elementwise_affine);

  check_input_parallel_dim_degrees(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> axis_dim_idxs = 
    transform(attrs.axes,
              [&](ff_dim_t d) -> parallel_tensor_dim_idx_t {
                return shard_dim_idx(d); 
              });

  std::set<parallel_tensor_dim_idx_t> map_to_sum_dim = {discard_copy_dim_idx()};

  std::set<parallel_tensor_dim_idx_t> map_to_discard_copy_dim = 
    set_minus(
      get_parallel_tensor_dim_indices(input_degrees),
      set_union(axis_dim_idxs, map_to_sum_dim));

  ParallelTensorDimDegrees result = ParallelTensorDimDegrees{
    /*sum_degree=*/SumDegree{
      dim_domain_get_volume(
        dim_domain_for_ptensor_dims(
          input_degrees,
          map_to_sum_dim)),
    },
    /*discard_copy_degree=*/DiscardCopyDegree{
      dim_domain_get_volume(
        dim_domain_for_ptensor_dims(
          input_degrees,
          map_to_discard_copy_dim)),
    },
    /*shard_degrees=*/
      ff_ordered_restrict_dims_strict(
        input_degrees.shard_degrees,
        attrs.axes),
  };

  ASSERT(
    get_total_degree_of_ptensor_dim_degrees(input_degrees) 
    == 
    get_total_degree_of_ptensor_dim_degrees(result)
  );

  return result;
}

ParallelTensorDimDegrees
    layer_norm_get_beta_weights_parallel_dim_degrees(LayerNormAttrs const &attrs, 
                                                     ParallelTensorDimDegrees const &input_degrees)
{
  ASSERT(attrs.elementwise_affine);

  check_input_parallel_dim_degrees(attrs, input_degrees);

  std::set<parallel_tensor_dim_idx_t> axis_dim_idxs = 
    transform(attrs.axes,
              [&](ff_dim_t d) -> parallel_tensor_dim_idx_t {
                return shard_dim_idx(d); 
              });

  std::set<parallel_tensor_dim_idx_t> map_to_sum_dim = {sum_dim_idx(), discard_copy_dim_idx()};

  std::set<parallel_tensor_dim_idx_t> map_to_discard_copy_dim = 
    set_minus(
      get_parallel_tensor_dim_indices(input_degrees),
      set_union(axis_dim_idxs, map_to_sum_dim));

  ParallelTensorDimDegrees result = ParallelTensorDimDegrees{
    /*sum_degree=*/SumDegree{
      dim_domain_get_volume(
        dim_domain_for_ptensor_dims(
          input_degrees,
          map_to_sum_dim)),
    },
    /*discard_copy_degree=*/DiscardCopyDegree{
      dim_domain_get_volume(
        dim_domain_for_ptensor_dims(
          input_degrees,
          map_to_discard_copy_dim)),
    },
    /*shard_degrees=*/
      ff_ordered_restrict_dims_strict(
        input_degrees.shard_degrees,
        attrs.axes),
  };

  ASSERT(
    get_total_degree_of_ptensor_dim_degrees(input_degrees) 
    == 
    get_total_degree_of_ptensor_dim_degrees(result)
  );

  return result;
}

std::map<TensorSlotName, ParallelTensorDimDegrees>
    layer_norm_get_weight_parallel_dim_degrees(LayerNormAttrs const &attrs,
                                               ParallelTensorDimDegrees const &input_shape)
{
  if (attrs.elementwise_affine) {
    return {
      {
        TensorSlotName::GAMMA,
        layer_norm_get_gamma_weights_parallel_dim_degrees(attrs, input_shape),
      },
      {
        TensorSlotName::BETA,
        layer_norm_get_beta_weights_parallel_dim_degrees(attrs, input_shape),
      },
    };
  } else {
    return {};
  }
}

ParallelTensorShape
    layer_norm_get_output_parallel_shape(LayerNormAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {

  TensorShape output_shape =
      layer_norm_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      layer_norm_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(output_shape, output_degrees);
}

ParallelTensorShape
    layer_norm_get_gamma_weights_parallel_shape(LayerNormAttrs const &attrs,
                                                ParallelTensorShape const &input_shape) {

  TensorShape gamma_shape =
      layer_norm_get_gamma_weights_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees gamma_degrees =
      layer_norm_get_gamma_weights_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(gamma_shape, gamma_degrees);
}

ParallelTensorShape
    layer_norm_get_beta_weights_parallel_shape(LayerNormAttrs const &attrs,
                           ParallelTensorShape const &input_shape) {

  TensorShape beta_shape =
      layer_norm_get_beta_weights_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees beta_degrees =
      layer_norm_get_beta_weights_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(beta_shape, beta_degrees);
}

std::map<TensorSlotName, ParallelTensorShape>
    layer_norm_get_weight_parallel_shapes(LayerNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {

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

StandardOperatorTaskGroup layer_norm_get_task_group(
    LayerNormAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    layer_norm_get_output_parallel_dim_degrees(attrs, input_degrees);

  StandardOperatorTaskGroup task_group = StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        std::set<parallel_tensor_dim_idx_t> axes_dims = 
          transform(attrs.axes,
                    [](ff_dim_t d) -> parallel_tensor_dim_idx_t {
                      return shard_dim_idx(d); 
                    });

        std::set<parallel_tensor_dim_idx_t> preexisting_input_sum_parallelism_dims = {sum_dim_idx()};

        std::set<parallel_tensor_dim_idx_t> weight_sum_parallelism_dims = {discard_copy_dim_idx()};

        std::set<parallel_tensor_dim_idx_t> remaining_dims = 
          set_minus(get_parallel_tensor_dim_indices(input_degrees),
                    set_union(std::vector{
                                axes_dims, 
                                weight_sum_parallelism_dims,
                                preexisting_input_sum_parallelism_dims,
                              }));

        OrthotopeBoundedCoord preexisting_input_sum_parallelism_coord = 
          orthotope_bounded_coord_for_ptensor_dims(
            input_degrees,
            input_coord,
            preexisting_input_sum_parallelism_dims);

        OrthotopeBoundedCoord weight_sum_parallelism_coord = 
          orthotope_bounded_coord_for_ptensor_dims(
            input_degrees,
            input_coord,
            weight_sum_parallelism_dims);

        OrthotopeBoundedCoord axes_coord = 
          orthotope_bounded_coord_for_ptensor_dims(
            input_degrees,
            input_coord,
            axes_dims);

        OrthotopeBoundedCoord remaining_dims_coord = 
          orthotope_bounded_coord_for_ptensor_dims(
            input_degrees,
            input_coord,
            remaining_dims);

        ParallelTensorSpaceCoordinate gamma_coord = 
          parallel_tensor_space_coordinate_from_bounded_orthotope_components(
            /*sum_component=*/assert_unwrap(
              flatten_orthotope_bounded_coord(
                weight_sum_parallelism_coord)),
            /*discard_copy_component=*/assert_unwrap(
              flatten_orthotope_bounded_coord(
                orthotope_bounded_coord_product(
                  remaining_dims_coord,
                  preexisting_input_sum_parallelism_coord))),
            /*shard_components=*/axes_coord);

        ParallelTensorSpaceCoordinate beta_coord =
          parallel_tensor_space_coordinate_from_bounded_orthotope_components(
            /*sum_component=*/assert_unwrap(
                flatten_orthotope_bounded_coord(
                  orthotope_bounded_coord_product(
                    weight_sum_parallelism_coord,
                    preexisting_input_sum_parallelism_coord))),
            /*discard_copy_component=*/assert_unwrap(
              flatten_orthotope_bounded_coord(
                remaining_dims_coord)),
            /*shard_components=*/axes_coord);

        ParallelTensorSpaceCoordinate output_coord = 
          ParallelTensorSpaceCoordinate{
            /*sum_component=*/assert_unwrap(
              flatten_orthotope_bounded_coord(
                orthotope_bounded_coord_product(
                  weight_sum_parallelism_coord,
                  preexisting_input_sum_parallelism_coord))).component,
            /*discard_copy_component=*/0_n,
            /*shard_components=*/input_coord.shard_components,
          };

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
            {
              TensorSlotName::INPUT,
              input_coord,
            },
            {
              TensorSlotName::GAMMA,
              gamma_coord,
            },
            {
              TensorSlotName::BETA,
              beta_coord,
            },
            {
              TensorSlotName::OUTPUT,
              output_coord,
            },
          },
          /*task_coord=*/task_coord_matching_parallel_tensor_space_coordinate(output_coord,
                                                                              output_degrees),
        };
      }),
  };

  return restrict_standard_operator_task_group_to_slots(
    task_group,
    layer_norm_get_slots(attrs));
}

ShardSignatureInstance
    layer_norm_get_shard_signature_instance(
          LayerNormAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    layer_norm_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace layer_norm_get_operator_task_space(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    layer_norm_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_input_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    layer_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_gamma_weights_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    layer_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::GAMMA);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_beta_weights_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ASSERT(attrs.elementwise_affine);

  StandardOperatorTaskGroup op_task_group =
    layer_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::BETA);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping layer_norm_get_operator_to_output_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    layer_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
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
