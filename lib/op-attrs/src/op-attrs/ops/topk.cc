#include "op-attrs/ops/topk.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape topk_get_output_shape(TopKAttrs const &, TensorShape const &) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorDimDegrees
    topk_get_output_parallel_dim_degrees(TopKAttrs const &,
                                         ParallelTensorDimDegrees const &)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}


ParallelTensorShape topk_get_output_parallel_shape(TopKAttrs const &attrs,
                                                   ParallelTensorShape const &input_shape)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorTaskSpace topk_get_operator_task_space(
    TopKAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping topk_get_operator_to_input_mapping(
    TopKAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping topk_get_operator_to_output_mapping(
    TopKAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
