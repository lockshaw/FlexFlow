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

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    embedding_get_input_to_output_mapping(EmbeddingAttrs const &attrs,
                                          ParallelTensorDimDegrees const &input_degrees) {

  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      inp_to_out = make_empty_eq_projection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>();

  num_tensor_dims_t input_num_dims = get_ptensor_dim_degrees_num_tensor_dims(input_degrees);

  ParallelTensorDimDegrees output_degrees =
      embedding_get_output_parallel_dim_degrees(attrs, input_degrees);

  num_tensor_dims_t output_num_dims = get_ptensor_dim_degrees_num_tensor_dims(output_degrees);

  parallel_tensor_dim_idx_t input_inner_dim =
    shard_dim_idx(ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, input_num_dims));

  parallel_tensor_dim_idx_t output_inner_dim =
    shard_dim_idx(ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, output_num_dims));

  project_dims(inp_to_out, input_inner_dim, shard_dim_idx());
  project_dims(inp_to_out, discard_copy_dim_idx(), output_inner_dim);
  for (ff_dim_t const &d :
       slice(tensor_dims_range(input_num_dims), 0, -1)) {
    project_dims(inp_to_output, d, d);
  }

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{inp_to_out}, input_degrees, output_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    embedding_get_weights_to_output_mapping(EmbeddingAttrs const &attrs,
                                            ParallelTensorDimDegrees const &input_degrees) {

  EqProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      weights_to_out = make_empty_eq_projection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>();

  ParallelTensorDimDegrees weights_degrees =
      embedding_get_weights_parallel_dim_degrees(attrs, input_degrees);

  ParallelTensorDimDegrees output_degrees =
      embedding_get_output_parallel_dim_degrees(attrs, input_degrees);

  num_tensor_dims_t output_num_dims = get_ptensor_dim_degrees_num_tensor_dims(output_degrees);

  parallel_tensor_dim_idx_t weights_entries_dim =
    shard_dim_idx(ff_dim_t{0_n});

  parallel_tensor_dim_idx_t weights_channel_dim =
    shard_dim_idx(ff_dim_t{1_n});

  parallel_tensor_dim_idx_t output_channel_dim =
    shard_dim_idx(ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, output_num_dims));

  project_dims(weights_to_out, discard_copy_dim_idx(), sum_dim_idx());
  project_dims(weights_to_out, weights_channel_dim, output_channel_dim);

  for (ff_dim_t const &d :
       slice(tensor_dims_range(output_num_dims), 0, -1)) {
    project_dims(weights_to_output, discard_copy_dim_idx(), d);
  }

  return parallel_tensor_space_biunique_mapping_from_projection(
      DimProjection{weights_to_out}, weights_degrees, output_degrees);
}

OperatorTaskSpace embedding_get_operator_task_space(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
      embedding_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_input_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping inp_to_out =
      embedding_get_input_to_output_mapping(attrs, input_degrees);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_inp =
      invert_parallel_tensor_space_biunique_mapping(inp_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      embedding_get_operator_to_output_mapping(attrs, input_degrees);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_inp);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_weights_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping weights_to_out =
      embedding_get_weights_to_output_mapping(attrs, input_degrees);

  ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping out_to_weights =
      invert_parallel_tensor_space_biunique_mapping(weights_to_out);

  OperatorSpaceToParallelTensorSpaceBiuniqueMapping op_to_out =
      embedding_get_operator_to_output_mapping(attrs, input_degrees);

  return operator_ptensor_space_biunique_mapping_from_composition(op_to_out, out_to_weights);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping embedding_get_operator_to_output_mapping(
    EmbeddingAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
      embedding_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_biunique_mapping(
      embedding_get_operator_task_space(attrs, input_degrees),
      output_degrees);
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
