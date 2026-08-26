#include "op-attrs/ops/dropout.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape dropout_get_output_shape(DropoutAttrs const &,
                             TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorDimDegrees
    dropout_get_output_parallel_dim_degrees(DropoutAttrs const &, ParallelTensorDimDegrees const &)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorShape
    dropout_get_output_parallel_shape(DropoutAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  ASSERT(
    get_sum_degree(input_shape) == 1,
    fmt::format("Expected sum degree 1, but receieved sum degree {}",
                get_sum_degree(input_shape))
  );

  ASSERT(
    get_discard_copy_degree(input_shape) == 1,
    fmt::format(
        "Expected discard copy degree 1, but received discard copy degree {}",
        get_discard_copy_degree(input_shape))
  );

  return input_shape;
}

OperatorTaskSpace dropout_get_operator_task_space(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping dropout_get_operator_to_input_mapping(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping dropout_get_operator_to_output_mapping(
    DropoutAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
