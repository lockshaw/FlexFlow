#include "op-attrs/ops/reduce.h"
#include "utils/exception.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReduceAttrs const &,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
