#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

TensorShape
    softmax_get_output_shape(SoftmaxAttrs const &attrs,
                     TensorShape const &input_shape) {

  ASSERT(
    attrs.dim.value < get_num_dims(input_shape.dims),
    fmt::format("get_output_shape for Softmax received out-of-bounds "
                "attrs.dim {} for input tensor shape {}",
                attrs.dim,
                input_shape)
  );

  return input_shape;
}

ParallelTensorShape
    softmax_get_output_parallel_shape(SoftmaxAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  TensorShape unpar = softmax_get_output_shape(attrs, get_reduced_shape(input_shape));

  ASSERT(
    get_sum_degree(input_shape) == 1,
    fmt::format("Expected sum degree 1, but received sum degree {}",
                get_sum_degree(input_shape))
  );

  ASSERT(
    get_discard_copy_degree(input_shape) == 1,
    fmt::format(
        "Expected discard copy degree 1, but received discard copy degree {}",
        get_discard_copy_degree(input_shape))
  );

  ASSERT(
    shard_dim_at_idx(input_shape, relative_ff_dim_t_from_ff_dim_t(attrs.dim)).degree == 1,
    fmt::format("Expected parallel degree of Softmax dimension {} to be 1, "
                "but received input shape {}",
                attrs.dim,
                input_shape)
  );

  return input_shape;
}

} // namespace FlexFlow
