#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/dim_ordered/get_idxs.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/range.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::unordered_map<parallel_tensor_dim_idx_t, int>
  get_parallel_tensor_degree_map(ParallelTensorDimDegrees const &degrees) {

  std::unordered_map<parallel_tensor_dim_idx_t, int> replica_dim_degrees = {
    {parallel_tensor_dim_idx_t{ReplicaType::SUM}, degrees.sum_degree.value},
    {parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}, degrees.discard_copy_degree.value},
  };

  std::unordered_map<ff_dim_t, int> shard_dim_degrees = 
    generate_map(get_idxs(degrees.shard_degrees),
                 [&](ff_dim_t const &dim) { return degrees.shard_degrees.at(dim); });

  return merge_maps(
    replica_dim_degrees,
    map_keys(shard_dim_degrees, [](ff_dim_t const &dim) { return parallel_tensor_dim_idx_t{dim}; }));
}

std::unordered_set<ParallelTensorSpaceCoordinate> 
  get_parallel_tensor_space_coordinates(ParallelTensorDimDegrees const &degrees) {

  std::unordered_map<parallel_tensor_dim_idx_t, int> degree_map = get_parallel_tensor_degree_map(degrees);

  std::unordered_map<
    parallel_tensor_dim_idx_t, 
    std::unordered_set<int>> possible_per_dim_coords
    = map_values(degree_map, [](int degree) { return unordered_set_of(range(degree)); });

  return transform(get_all_assignments(possible_per_dim_coords),
                   [](std::unordered_map<parallel_tensor_dim_idx_t, int> const &m) {
                     return parallel_tensor_space_coord_from_map(m);
                   });
}

} // namespace FlexFlow
