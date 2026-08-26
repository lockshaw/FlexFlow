#include "op-attrs/ops/softmax.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/not_implemented.h"

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

ParallelTensorDimDegrees
    softmax_get_output_parallel_dim_degrees(SoftmaxAttrs const &attrs,
                     ParallelTensorDimDegrees const &input_shape)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
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

OperatorTaskSpace softmax_get_operator_task_space(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping softmax_get_operator_to_input_mapping(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping softmax_get_operator_to_output_mapping(
    SoftmaxAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
