#include "op-attrs/ops/repartition.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
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

ParallelismOperatorTaskGroup
    repartition_get_task_group(RepartitionAttrs const &attrs,
                               ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      repartition_get_output_parallel_dim_degrees(attrs, input_degrees);

  return ParallelismOperatorTaskGroup{
      transform(get_parallel_tensor_space_coordinates(output_degrees),
                [&](ParallelTensorSpaceCoordinate const &output_coord)
                    -> AbstractedOperatorAtomicTaskShardBinding {
                  ParallelTensorSpaceCoordinate input_coord = output_coord;
                  ptensor_coord_component_for_ptensor_dim_idx(
                      input_coord, shard_dim_idx(attrs.repartition_dim)) /=
                      attrs.repartition_degree;

                  return AbstractedOperatorAtomicTaskShardBinding{
                      /*tensor_coords=*/{
                          {
                              TensorSlotName::INPUT,
                              input_coord,
                          },
                          {
                              TensorSlotName::OUTPUT,
                              output_coord,
                          },
                      },
                      /*task_coord=*/
                      task_coord_matching_parallel_tensor_space_coordinate(
                          output_coord, output_degrees),
                  };
                }),
  };
}

} // namespace FlexFlow
