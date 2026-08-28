#include "op-attrs/ops/noop.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"

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
  ParallelTensorDimDegrees output_degrees =
      noop_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping noop_get_operator_to_input_mapping(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  return get_identity_biunique_mapping(
      noop_get_operator_task_space(attrs, input_degrees),
      input_degrees);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping noop_get_operator_to_output_mapping(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
      noop_get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_biunique_mapping(
      noop_get_operator_task_space(attrs, input_degrees),
      output_degrees);
}

} // namespace FlexFlow
