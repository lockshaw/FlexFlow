#include "op-attrs/ops/gather.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape gather_get_output_shape(GatherAttrs const &,
                                    TensorShape const &input,
                                    TensorShape const &index) {
  NOT_IMPLEMENTED();
}

ParallelTensorDimDegrees gather_get_output_parallel_dim_degrees(
    GatherAttrs const &,
    ParallelTensorDimDegrees const &input,
    ParallelTensorDimDegrees const &index) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape
    gather_get_output_parallel_shape(GatherAttrs const &,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
