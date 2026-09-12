#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers/require_same.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"

namespace FlexFlow {

MultiHeadAttentionParallelInputs
    parse_attention_parallel_input_shape(ParallelTensorDimDegrees const &input_q,
                                         ParallelTensorDimDegrees const &input_k,
                                         ParallelTensorDimDegrees const &input_v) {
  num_ptensor_shard_dims_t input_num_shard_dims =
    require_same(get_ptensor_dim_degrees_num_shard_dims(input_q),
                 get_ptensor_dim_degrees_num_shard_dims(input_k),
                 get_ptensor_dim_degrees_num_shard_dims(input_v));

  ASSERT(input_num_shard_dims.value == 3);

  relative_ff_dim_t seq_dim = relative_ff_dim_t{-2};

  positive_int seq_len_degree =
    require_same(
      get_degree_for_relative_ff_dim_t(input_q, seq_dim),
      get_degree_for_relative_ff_dim_t(input_k, seq_dim),
      get_degree_for_relative_ff_dim_t(input_v, seq_dim));

  ASSERT(seq_len_degree == 1);

  relative_ff_dim_t batch_dim = relative_ff_dim_t{-3};

  positive_int batch_degree =
    require_same(
      get_degree_for_relative_ff_dim_t(input_q, batch_dim),
      get_degree_for_relative_ff_dim_t(input_k, batch_dim),
      get_degree_for_relative_ff_dim_t(input_v, batch_dim));

  positive_int query_degree = get_degree_for_relative_ff_dim_t(input_q, relative_ff_dim_t{-1});
  ASSERT(
    query_degree == 1,
    fmt::format("Expected query tensor to have query dim parallel degree "
                "1, but received degree {}",
                query_degree)
  );

  positive_int key_degree = get_degree_for_relative_ff_dim_t(input_k, relative_ff_dim_t{-1});
  ASSERT(
    key_degree == 1,
    fmt::format("Expected key tensor to have key dim parallel degree 1, "
                "but received degree {}",
                key_degree)
  );

  positive_int value_degree = get_degree_for_relative_ff_dim_t(input_v, relative_ff_dim_t{-1});
  ASSERT(
    value_degree == 1,
    fmt::format("Expected value tensor to have value dim parallel degree "
                "1, but received degree {}",
                value_degree)
  );

  DiscardCopyDegree discard_copy_degree =
    require_same(
      input_q.discard_copy_degree,
      input_k.discard_copy_degree,
      input_v.discard_copy_degree);

  return MultiHeadAttentionParallelInputs{
      /*batch_degree=*/batch_degree,
      /*sequence_degree=*/seq_len_degree,
      /*query_degree=*/query_degree,
      /*key_degree=*/key_degree,
      /*value_degree=*/value_degree,
      /*discard_copy_degree=*/discard_copy_degree,
  };
}

} // namespace FlexFlow
