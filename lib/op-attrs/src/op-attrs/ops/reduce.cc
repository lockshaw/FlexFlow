#include "op-attrs/ops/reduce.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape reduce_get_output_shape(ReduceAttrs const &,
                                    TensorShape const &)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorDimDegrees reduce_get_output_parallel_dim_degrees(
          ReduceAttrs const &,
          ParallelTensorDimDegrees const &)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorShape reduce_get_output_parallel_shape(ReduceAttrs const &,
                                                     ParallelTensorShape const &)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorTaskSpace reduce_get_operator_task_space(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reduce_get_operator_to_input_mapping(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reduce_get_operator_to_output_mapping(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
