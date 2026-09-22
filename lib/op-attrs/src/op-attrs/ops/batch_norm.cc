#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/any_of.h"
#include "utils/containers/extend.h"
#include "utils/expected.h"
#include "utils/containers/all_of.h"
#include "utils/orthotope/bounded_component.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include "utils/optional.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_batch_norm_incoming_tensor_roles(BatchNormAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {
          TensorSlotName::INPUT,
          IncomingTensorRole::INPUT,
      },
  };

  if (attrs.affine) {
    result[TensorSlotName::GAMMA] = IncomingTensorRole::WEIGHT;
    result[TensorSlotName::BETA] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

void check_input_shape(BatchNormAttrs const &, TensorShape const &input_shape) {

  ASSERT(
    get_num_dims(input_shape.dims) >= 2,
    fmt::format(
        "BatchNormAttrs expected input dims >= 2, but received input shape {}",
        input_shape)
  );

  ASSERT(
    input_shape.data_type == DataType::FLOAT,
    fmt::format("BatchNormAttrs currently only supports data_type = "
                "FLOAT, but received input data_type {}. "
                "If you need this feature, please create an issue.",
                input_shape.data_type)
  );
}

TensorShape
    batch_norm_get_output_shape(BatchNormAttrs const &attrs,
                     TensorShape const &input_shape) {

  check_input_shape(attrs, input_shape);

  return input_shape;
}

TensorShape
    batch_norm_get_gamma_weights_shape(BatchNormAttrs const &attrs,
                            TensorShape const &input_shape) {
  check_input_shape(attrs, input_shape);

  ASSERT(attrs.affine, "No gamma weights exist for attrs.affine = false");

  positive_int num_channels =
      dim_at_idx(input_shape.dims, relative_ff_dim_t{1});

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          num_channels,
      }},
      DataType::FLOAT,
  };
}

TensorShape
    batch_norm_get_beta_weights_shape(BatchNormAttrs const &attrs,
                           TensorShape const &input_shape) {

  ASSERT(attrs.affine, "No beta weights exist for attrs.affine = false");

  return batch_norm_get_gamma_weights_shape(attrs, input_shape);
}

std::map<TensorSlotName, TensorShape>
    batch_norm_get_weight_shapes(BatchNormAttrs const &attrs,
                      TensorShape const &input_shape) {

  TensorShape gamma_shape =
      batch_norm_get_gamma_weights_shape(attrs, input_shape);
  TensorShape beta_shape =
      batch_norm_get_beta_weights_shape(attrs, input_shape);

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
    check_input_degrees(BatchNormAttrs const &,
                        ParallelTensorDimDegrees const &input_degrees) {

  ASSERT(
    input_degrees.shard_degrees.size() >= 2,
    fmt::format("BatchNormAttrs expected input dims >= 2, but received "
                       "input degrees {}",
                       input_degrees)
  );

  ASSERT(
    input_degrees.sum_degree == SumDegree{1_p},
    fmt::format("Expected sum degree 1, but receieved sum degree {}",
                input_degrees.sum_degree)
  );
}

ParallelTensorDimDegrees
    batch_norm_get_output_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  check_input_degrees(attrs, input_degrees);

  return input_degrees;
}

ParallelTensorDimDegrees
    batch_norm_get_gamma_weights_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  check_input_degrees(attrs, input_degrees);

  ASSERT(attrs.affine, "No gamma weights exist for attrs.affine = false");

  relative_ff_dim_t channel_dim = relative_ff_dim_t{1};

  return ParallelTensorDimDegrees{
      SumDegree{1_p},
      DiscardCopyDegree{1_p},
      FFOrdered<positive_int>{input_degrees.shard_degrees.at(channel_dim)},
  };
}

ParallelTensorDimDegrees
    batch_norm_get_beta_weights_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  check_input_degrees(attrs, input_degrees);

  ASSERT(attrs.affine, "No beta weights exist for attrs.affine = false");

  return batch_norm_get_gamma_weights_parallel_dim_degrees(attrs, input_degrees);
}

std::map<TensorSlotName, ParallelTensorDimDegrees>
    batch_norm_get_weight_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees gamma_degrees =
      batch_norm_get_gamma_weights_parallel_dim_degrees(attrs, input_degrees);
  ParallelTensorDimDegrees beta_degrees =
      batch_norm_get_beta_weights_parallel_dim_degrees(attrs, input_degrees);

  return std::map<TensorSlotName, ParallelTensorDimDegrees>{
      {
          TensorSlotName::GAMMA,
          gamma_degrees,
      },
      {
          TensorSlotName::BETA,
          beta_degrees,
      },
  };
}

ParallelTensorShape
    batch_norm_get_output_parallel_shape(BatchNormAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  TensorShape unpar = batch_norm_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = batch_norm_get_output_parallel_dim_degrees(attrs,
                                        get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

ParallelTensorShape
    batch_norm_get_gamma_weights_parallel_shape(BatchNormAttrs const &attrs,
                            ParallelTensorShape const &input_shape) {

  TensorShape unpar = batch_norm_get_gamma_weights_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = batch_norm_get_gamma_weights_parallel_dim_degrees(
            attrs, get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

ParallelTensorShape
    batch_norm_get_beta_weights_parallel_shape(BatchNormAttrs const &attrs,
                           ParallelTensorShape const &input_shape) {

  TensorShape unpar = batch_norm_get_beta_weights_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = batch_norm_get_beta_weights_parallel_dim_degrees(
            attrs, get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

std::map<TensorSlotName, ParallelTensorShape>
    batch_norm_get_weight_parallel_shapes(BatchNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {

  ParallelTensorShape gamma_shape =
      batch_norm_get_gamma_weights_parallel_shape(attrs, input_shape);
  ParallelTensorShape beta_shape =
      batch_norm_get_beta_weights_parallel_shape(attrs, input_shape);

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

StandardOperatorTaskGroup batch_norm_get_task_group(
    BatchNormAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    batch_norm_get_output_parallel_dim_degrees(attrs, input_degrees);

  return StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        parallel_tensor_dim_idx_t channel_dim = shard_dim_idx(ff_dim_t{1_n});

        std::set<parallel_tensor_dim_idx_t> non_channel_dims =
          set_minus(
            get_parallel_tensor_dim_indices(input_degrees),
            std::set{channel_dim});

        BoundedComponent channel_parallelism_component =
          bounded_component_for_ptensor_dim(
            input_degrees,
            input_coord,
            channel_dim);

        OrthotopeBoundedCoord non_channel_parallelism_components =
          orthotope_bounded_coord_for_ptensor_dims(
            input_degrees,
            input_coord,
            non_channel_dims);

        ParallelTensorSpaceCoordinate weight_coord =
              parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                /*sum_degree=*/trivial_bounded_component(),
                /*discard_copy_degree=*/assert_unwrap(
                    flatten_orthotope_bounded_coord(non_channel_parallelism_components)),
                /*shard_coords=*/lift_bounded_component(channel_parallelism_component));

        ParallelTensorSpaceCoordinate output_coord = input_coord;

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
            {
              TensorSlotName::INPUT,
              input_coord,
            },
            {
              TensorSlotName::GAMMA,
              weight_coord,
            },
            {
              TensorSlotName::BETA,
              weight_coord,
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
}

ShardSignatureInstance
    batch_norm_get_shard_signature_instance(
          BatchNormAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    batch_norm_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace batch_norm_get_operator_task_space(
    BatchNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    batch_norm_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping batch_norm_get_operator_to_input_mapping(
    BatchNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    batch_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    batch_norm_get_operator_to_gamma_weights_mapping(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    batch_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::GAMMA);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping batch_norm_get_operator_to_beta_weights_mapping(
    BatchNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    batch_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::BETA);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping batch_norm_get_operator_to_output_mapping(
    BatchNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    batch_norm_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

std::map<TensorSlotName, InitializerAttrs>
    batch_norm_get_initializers(BatchNormAttrs const &attrs) {
  if (attrs.affine) {
    InitializerAttrs gamma_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{1}}}};

    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};

    return std::map<TensorSlotName, InitializerAttrs>{
        {
            TensorSlotName::GAMMA,
            gamma_initializer,
        },
        {
            TensorSlotName::BETA,
            beta_initializer,
        },
    };
  } else {
    return std::map<TensorSlotName, InitializerAttrs>{};
  }
}

} // namespace FlexFlow
