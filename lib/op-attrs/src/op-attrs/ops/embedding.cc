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
