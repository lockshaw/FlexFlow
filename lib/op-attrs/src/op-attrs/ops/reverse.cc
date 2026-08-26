#include "op-attrs/ops/reverse.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"

namespace FlexFlow {


TensorShape reverse_get_output_shape(ReverseAttrs const &attrs, TensorShape const &input_shape)
{
  ASSERT(
    tensor_dims_has_dim(input_shape.dims, attrs.axis)
  );

  return input_shape;
}

ParallelTensorDimDegrees
    reverse_get_output_parallel_dim_degrees(ReverseAttrs const &attrs,
                                            ParallelTensorDimDegrees const &input_dim_degrees)
{
  ASSERT(
    get_degree_for_parallel_tensor_dim_idx(input_dim_degrees, shard_dim_idx(attrs.axis)) == 1_p
  );

  return input_dim_degrees;
}

ParallelTensorShape reverse_get_output_parallel_shape(ReverseAttrs const &attrs,
                                                      ParallelTensorShape const &input_shape)
{
  TensorShape unpar = reverse_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees degrees = reverse_get_output_parallel_dim_degrees(attrs,
                                        get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

OperatorTaskSpace reverse_get_operator_task_space(
    ReverseAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reverse_get_operator_to_input_mapping(
    ReverseAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reverse_get_operator_to_output_mapping(
    ReverseAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
