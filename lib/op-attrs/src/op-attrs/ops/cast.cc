#include "op-attrs/ops/cast.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_dims.h"

namespace FlexFlow {

TensorShape
    cast_get_output_shape(CastAttrs const &attrs, TensorShape const &input) {

  TensorShape output = input;
  output.data_type = attrs.dtype;
  return output;
}

ParallelTensorDimDegrees cast_get_output_parallel_dim_degrees(
  CastAttrs const &attrs, ParallelTensorDimDegrees const &input_dim_degrees) {

  return input_dim_degrees;
}

ParallelTensorShape
    cast_get_output_parallel_shape(CastAttrs const &attrs, ParallelTensorShape const &input) {

  TensorShape unpar = cast_get_output_shape(attrs, get_reduced_shape(input));

  ParallelTensorDimDegrees output_degrees =
      cast_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_to_parallel_with_degrees(unpar, output_degrees);
}

OperatorTaskSpace cast_get_operator_task_space(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping cast_get_operator_to_input_mapping(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping cast_get_operator_to_output_mapping(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
