#include "op-attrs/ops/noop.h"

namespace FlexFlow {

TensorShape noop_get_output_shape(NoopAttrs const &,
                             TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorDimDegrees noop_get_output_parallel_dim_degrees(NoopAttrs const &,
                                                              ParallelTensorDimDegrees const &input_dim_degrees)
{
  return input_dim_degrees;
}

ParallelTensorShape noop_get_output_parallel_shape(NoopAttrs const &,
                                                   ParallelTensorShape const &input_shape) {
  return input_shape;
}

OperatorTaskSpace noop_get_operator_task_space(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping noop_get_operator_to_input_mapping(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping noop_get_operator_to_output_mapping(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
