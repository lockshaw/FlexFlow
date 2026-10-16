#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"

namespace FlexFlow {

ParallelTensorShape replicate_get_output_parallel_shape(
    ReplicateAttrs const &attrs, ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.replica_dims.discard_copy_degree.value *=
      attrs.replicate_degree;
  return output_shape;
}

ParallelTensorDimDegrees replicate_get_output_parallel_dim_degrees(
    ReplicateAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.discard_copy_degree.value *= attrs.replicate_degree;
  return output_degrees;
}

} // namespace FlexFlow
