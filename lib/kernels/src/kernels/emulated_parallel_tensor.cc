#include "kernels/emulated_parallel_tensor.h"
#include "kernels/accessors_are_equal.h"
#include "utils/containers/all_of.h"
#include "kernels/create_random_filled_accessor.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/containers/generate_map.h"

namespace FlexFlow {

EmulatedParallelTensor random_emulated_parallel_tensor_of_shape(ParallelTensorShape const &pt_shape,
                                                                Allocator &allocator,
                                                                int seed)
{
  std::mt19937 gen(seed);

  TensorShape shard_shape = get_piece_shape(pt_shape);

  auto without_discard_copy = [&](ParallelTensorSpaceCoordinate const &c) 
    -> ParallelTensorSpaceCoordinate
  {
    ParallelTensorSpaceCoordinate result = c;
    result.discard_copy_component = 0_n; 
    return result;
  };

  std::set<ParallelTensorSpaceCoordinate> points = 
      get_parallel_tensor_space_coordinates(get_parallel_degrees(pt_shape));

  std::set<ParallelTensorSpaceCoordinate> core_points = 
    transform(points, without_discard_copy);

  std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>
    core_shards = generate_map(
      core_points,
      [&](ParallelTensorSpaceCoordinate const &c) -> GenericTensorAccessorR {
        return create_random_filled_accessor_r_with_gen(
          /*shape=*/shard_shape, 
          /*allocator=*/allocator, 
          /*gen=*/gen);
      });

  return EmulatedParallelTensor{
    generate_map(
      points,
      [&](ParallelTensorSpaceCoordinate const &c) -> GenericTensorAccessorR {
        return core_shards.at(without_discard_copy(c));
      }),
  };
}

bool emulated_parallel_tensors_are_equal(EmulatedParallelTensor const &lhs,
                                         EmulatedParallelTensor const &rhs)
{
  std::set<ParallelTensorSpaceCoordinate> lhs_coords = keys(lhs.shards);
  std::set<ParallelTensorSpaceCoordinate> rhs_coords = keys(rhs.shards);

  if (lhs_coords != rhs_coords) {
    return false;
  }

  return all_of(
    require_same(lhs_coords, rhs_coords),
    [&](ParallelTensorSpaceCoordinate const &c) -> bool {
      return accessors_are_equal(lhs.shards.at(c), rhs.shards.at(c));
    });
}

} // namespace FlexFlow
