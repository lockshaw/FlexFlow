#include "op-attrs/ops/repartition.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

ParallelTensorShape repartition_get_output_parallel_shape(
    RepartitionAttrs const &attrs, ParallelTensorShape const &input_shape) {
  ASSERT(input_shape.dims.shard_dims.idx_is_valid(attrs.repartition_dim),
         attrs,
         input_shape);

  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.shard_dims
      .at(relative_ff_dim_t_from_ff_dim_t(attrs.repartition_dim))
      .degree *= attrs.repartition_degree;
  return output_shape;
}

ParallelTensorDimDegrees repartition_get_output_parallel_dim_degrees(
    RepartitionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.shard_degrees.at(relative_ff_dim_t_from_ff_dim_t(
      attrs.repartition_dim)) *= attrs.repartition_degree;
  return output_degrees;
}

} // namespace FlexFlow
