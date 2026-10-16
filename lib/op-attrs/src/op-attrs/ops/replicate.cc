#include "op-attrs/ops/replicate.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/task_space_coordinate.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"

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

ParallelismOperatorTaskGroup
    replicate_get_task_group(ReplicateAttrs const &attrs,
                             ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      replicate_get_output_parallel_dim_degrees(attrs, input_degrees);

  return ParallelismOperatorTaskGroup{
      transform(get_parallel_tensor_space_coordinates(output_degrees),
                [&](ParallelTensorSpaceCoordinate const &output_coord)
                    -> AbstractedOperatorAtomicTaskShardBinding {
                  ParallelTensorSpaceCoordinate input_coord = output_coord;
                  input_coord.discard_copy_component /= attrs.replicate_degree;

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
