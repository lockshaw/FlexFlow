#include "op-attrs/ops/topk.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape topk_get_output_shape(TopKAttrs const &, TensorShape const &) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorDimDegrees
    topk_get_output_parallel_dim_degrees(TopKAttrs const &,
                                         ParallelTensorDimDegrees const &) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorShape
    topk_get_output_parallel_shape(TopKAttrs const &attrs,
                                   ParallelTensorShape const &input_shape) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

StandardOperatorTaskGroup
    topk_get_task_group(TopKAttrs const &, ParallelTensorDimDegrees const &) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
