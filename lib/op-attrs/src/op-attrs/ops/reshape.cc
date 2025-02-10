#include "op-attrs/ops/reshape.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape get_output_shape(ReshapeAttrs const &attrs,
                             TensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(ReshapeAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
