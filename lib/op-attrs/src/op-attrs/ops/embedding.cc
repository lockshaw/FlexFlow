#include "op-attrs/ops/embedding.h"
#include "op-attrs/dim_ordered/slice.h"
#include "op-attrs/dim_ordered/transform.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

RecordFormatter as_dot(EmbeddingAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("num_entries", attrs.num_entries)
    << kv("out_channels", attrs.out_channels) << kv("aggr", attrs.aggr)
    << kv("output_type", attrs.data_type);

  return r;
}

static std::optional<std::string> basic_check(EmbeddingAttrs const &attrs,
                                              TensorShape const &input) {
  if (input.data_type != DataType::INT32 &&
      input.data_type != DataType::INT64) {
    return fmt::format("Embedding expected input tensor to have integer "
                       "datatype, but receieved tensor of datatype {}",
                       input.data_type);
  }

  if (attrs.aggr != AggregateOp::SUM) {
    return fmt::format(
        "Currently unsupported aggregation op for embedding: {}", attrs.aggr);
  }

  return std::nullopt;
}

tl::expected<TensorShape, std::string>
    get_output_shape(EmbeddingAttrs const &attrs, TensorShape const &input) {
  {
    std::optional<std::string> err_msg = basic_check(attrs, input);
    if (err_msg.has_value()) {
      return tl::unexpected(err_msg.value());
    }
  }

  TensorShape output = input;
  dim_at_idx(output, relative_ff_dim_t{-1}) = attrs.out_channels;
  output.data_type = attrs.data_type;
  return output;
}

tl::expected<TensorShape, std::string>
    get_weights_shape(EmbeddingAttrs const &attrs, TensorShape const &input) {
  {
    std::optional<std::string> err_msg = basic_check(attrs, input);
    if (err_msg.has_value()) {
      return tl::unexpected(err_msg.value());
    }
  }

  return TensorShape{
      TensorDims{
          FFOrdered<nonnegative_int>{
              attrs.num_entries,
              attrs.out_channels,
          },
      },
      attrs.data_type,
  };
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(EmbeddingAttrs const &attrs,
                     ParallelTensorShape const &input) {

  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree =
      SumDegree{shard_dim_at_idx(input, relative_ff_dim_t{-1}).degree};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_n};
  FFOrdered<nonnegative_int> shard_degrees =
      transform(input.dims.shard_dims,
                [](ShardParallelDim const &d) { return d.degree; });
  shard_degrees.at(relative_ff_dim_t{-1}) = get_discard_copy_degree(input);

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(EmbeddingAttrs const &attrs,
                      ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_weights_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1_n};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{product(transform(
      ff_ordered_shard_dims(input.dims),
      [](ShardParallelDim const &d) -> nonnegative_int { return d.degree; }))};
  nonnegative_int entry_dim_degree = 1_n;
  nonnegative_int out_channel_degree = get_discard_copy_degree(input);
  FFOrdered<nonnegative_int> shard_degrees = {
      entry_dim_degree,
      out_channel_degree,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

} // namespace FlexFlow
