#include "op-attrs/ops/embedding.h"
#include "op-attrs/ff_ordered/ff_ordered_slice.h"
#include "op-attrs/ff_ordered/ff_ordered_transform.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/orthotope/bounded_component.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include "op-attrs/task_space_coordinate.h"

namespace FlexFlow {

static void basic_check(EmbeddingAttrs const &attrs, TensorShape const &input) {
  ASSERT(
    input.data_type == DataType::INT32
    ||
    input.data_type == DataType::INT64,
    fmt::format("Embedding expected input tensor to have integer "
                "datatype, but receieved tensor of datatype {}",
                input.data_type)
  );

  ASSERT(
    attrs.aggr == AggregateOp::SUM,
    fmt::format("Currently unsupported aggregation op for embedding: {}",
                attrs.aggr)
  );
}

TensorShape
    embedding_get_output_shape(EmbeddingAttrs const &attrs, TensorShape const &input) {

  basic_check(attrs, input);

  TensorShape output = input;
  dim_at_idx(output.dims, relative_ff_dim_t{-1}) = attrs.out_channels;
  output.data_type = attrs.data_type;
  return output;
}

TensorShape
    embedding_get_weights_shape(EmbeddingAttrs const &attrs, TensorShape const &input) {

  basic_check(attrs, input);

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              attrs.num_entries,
              attrs.out_channels,
          },
      },
      attrs.data_type,
  };
}

ParallelTensorDimDegrees embedding_get_output_parallel_dim_degrees(
  EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_dim_degrees)
{
  SumDegree sum_degree = SumDegree{
    input_dim_degrees.shard_degrees.at(relative_ff_dim_t{-1}),
  };

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};
  FFOrdered<positive_int> shard_degrees = input_dim_degrees.shard_degrees;
  shard_degrees.at(relative_ff_dim_t{-1}) = input_dim_degrees.discard_copy_degree.value;

  return ParallelTensorDimDegrees{
    /*sum_degree=*/sum_degree,
    /*discard_copy_degree=*/discard_copy_degree,
    /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorDimDegrees embedding_get_weights_parallel_dim_degrees(
  EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_dim_degrees)
{
  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{product(input_dim_degrees.shard_degrees)};

  positive_int entry_dim_degree = 1_p;
  positive_int out_channel_degree = input_dim_degrees.discard_copy_degree.value;
  FFOrdered<positive_int> shard_degrees = FFOrdered{
      entry_dim_degree,
      out_channel_degree,
  };

  return ParallelTensorDimDegrees{
    /*sum_degree=*/sum_degree,
    /*discard_copy_degree=*/discard_copy_degree,
    /*shard_degrees=*/shard_degrees,
  };
}

ParallelTensorShape
    embedding_get_output_parallel_shape(EmbeddingAttrs const &attrs,
                                        ParallelTensorShape const &input) {

  TensorShape unpar = embedding_get_output_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees output_degrees =
      embedding_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_to_parallel_with_degrees(unpar, output_degrees);
}

ParallelTensorShape
    embedding_get_weights_parallel_shape(EmbeddingAttrs const &attrs,
                      ParallelTensorShape const &input) {
  TensorShape unpar = embedding_get_weights_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees weight_degrees =
      embedding_get_weights_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_to_parallel_with_degrees(unpar, weight_degrees);
}

StandardOperatorTaskGroup embedding_get_task_group(
    EmbeddingAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {

  num_ptensor_shard_dims_t input_num_shard_dims =
    get_ptensor_dim_degrees_num_shard_dims(input_degrees);

  ASSERT(input_degrees.sum_degree == SumDegree{1_p});

  return StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        parallel_tensor_dim_idx_t sum_dim = sum_dim_idx();
        parallel_tensor_dim_idx_t discard_copy_dim = discard_copy_dim_idx();

        std::set<parallel_tensor_dim_idx_t> leading_dims =
          shard_dim_idxs_for_exclusive_interval(0, -1, input_num_shard_dims);

        parallel_tensor_dim_idx_t onehot_dim =
          shard_dim_idx_for_relative(-1, input_num_shard_dims);

        OrthotopeBoundedCoord data_parallelism_coord =
          orthotope_bounded_coord_for_ptensor_dims(
            input_degrees,
            input_coord,
            leading_dims);

        BoundedComponent onehot_parallelism_coord =
            bounded_component_for_ptensor_dim(
              input_degrees,
              input_coord,
              onehot_dim);

        BoundedComponent output_channel_parallelism_coord =
            bounded_component_for_ptensor_dim(
              input_degrees,
              input_coord,
              discard_copy_dim);

        BoundedComponent output_copy_component =
          trivial_bounded_component();

        OrthotopeBoundedCoord output_shard_components =
                orthotope_bounded_coord_product(
                  data_parallelism_coord,
                  lift_bounded_component(output_channel_parallelism_coord));

        OrthotopeBoundedCoord raw_output_coord =
              orthotope_bounded_coord_product(
                lift_bounded_component(onehot_parallelism_coord),
                lift_bounded_component(output_copy_component),
                output_shard_components);

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_corods=*/{
            {
              TensorSlotName::INPUT,
              input_coord,
            },
            {
              TensorSlotName::WEIGHT,
              parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                /*sum_degree=*/trivial_bounded_component(),
                /*discard_copy_degree=*/onehot_parallelism_coord,
                /*shard_coords=*/
                  make_2d_orthotope_bounded_coord(
                    trivial_bounded_component(),
                    output_channel_parallelism_coord)),
            },
            {
              TensorSlotName::OUTPUT,
              parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                /*sum_degree=*/onehot_parallelism_coord,
                /*discard_copy_degree=*/output_copy_component,
                /*shard_coords=*/output_shard_components),
            },
          },
          /*task_coord=*/task_space_coordinate_from_orthotope_coord(raw_output_coord.coord),
        };
      }),
  };
}

OperatorTaskSpace embedding_get_operator_task_space(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    embedding_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

ShardSignatureInstance
    embedding_get_shard_signature_instance(EmbeddingAttrs const &attrs,
                                           ParallelTensorDimDegrees const &input_degrees) {

  StandardOperatorTaskGroup op_task_group =
    embedding_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_input_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    embedding_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_weights_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    embedding_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::WEIGHT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_output_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    embedding_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

std::map<TensorSlotName, InitializerAttrs> embedding_get_initializers(
    EmbeddingAttrs const &,
    std::optional<InitializerAttrs> const &maybe_initializer_attrs) {
  InitializerAttrs default_initializer_attrs = InitializerAttrs{
      NormInitializerAttrs{
          /*seed=*/0,
          /*mean=*/0.0,
          /*stddev=*/1.0,
      },
  };

  return {
      {
          TensorSlotName::WEIGHT,
          maybe_initializer_attrs.value_or(default_initializer_attrs),
      },
  };
}

} // namespace FlexFlow
