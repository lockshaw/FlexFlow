#include "op-attrs/ops/noop.h"

namespace FlexFlow {

TensorShape noop_get_output_shape(NoopAttrs const &,
                                  TensorShape const &input_shape) {
  return input_shape;
}

ParallelTensorDimDegrees noop_get_output_parallel_dim_degrees(
    NoopAttrs const &, ParallelTensorDimDegrees const &input_dim_degrees) {
  return input_dim_degrees;
}

ParallelTensorShape
    noop_get_output_parallel_shape(NoopAttrs const &,
                                   ParallelTensorShape const &input_shape) {
  return input_shape;
}

} // namespace FlexFlow
