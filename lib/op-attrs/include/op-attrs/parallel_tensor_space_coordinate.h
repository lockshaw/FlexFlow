#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_COORDINATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_COORDINATE_H

#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "utils/orthotope/dim_coord.dtg.h"

namespace FlexFlow {

num_ptensor_parallel_dims_t
    ptensor_coord_num_dims(ParallelTensorSpaceCoordinate const &);
num_ptensor_shard_dims_t
    ptensor_coord_num_shard_dims(ParallelTensorSpaceCoordinate const &);

std::set<parallel_tensor_dim_idx_t>
    get_dim_idxs_in_ptensor_space_coord(ParallelTensorSpaceCoordinate const &);

nonnegative_int ptensor_coord_component_for_ptensor_dim_idx(
    ParallelTensorSpaceCoordinate const &, parallel_tensor_dim_idx_t);

nonnegative_int &ptensor_coord_component_for_ptensor_dim_idx(
    ParallelTensorSpaceCoordinate &, parallel_tensor_dim_idx_t);

nonnegative_int
    flattened_component_for_ptensor_dims(ParallelTensorDimDegrees const &,
                                         ParallelTensorSpaceCoordinate const &,
                                         std::set<parallel_tensor_dim_idx_t> const &);

OrthotopeBoundedCoord
    orthotope_bounded_coord_for_ptensor_dims(ParallelTensorDimDegrees const &,
                                             ParallelTensorSpaceCoordinate const &,
                                             std::set<parallel_tensor_dim_idx_t> const &);

ParallelTensorSpaceCoordinate
  parallel_tensor_space_coordinate_from_bounded_orthotope_coords(
    BoundedOrthotopeCoord const &sum_coord,
    BoundedOrthotopeCoord const &discard_copy_coord,
    std::vector<BoundedOrthotopeCoord> const &shard_coords);

ParallelTensorSpaceCoordinate
  parallel_tensor_space_coordinate_from_ff_ordered(
    nonnegative_int const &sum_component,
    nonnegative_int const &discard_copy_component,
    std::vector<nonnegative_int> const &shard_components);

OrthotopeBoundedCoord
    orthotope_bounded_coord_for_ptensor_dims(ParallelTensorDimDegrees const &,
                                             ParallelTensorSpaceCoordinate const &,
                                             std::set<parallel_tensor_dim_idx_t> const &);

ParallelTensorSpaceCoordinate parallel_tensor_space_coord_from_map(
    std::map<parallel_tensor_dim_idx_t, nonnegative_int> const &);

ParallelTensorSpaceCoordinate parallel_tensor_space_coord_from_dim_coord(
    DimCoord<parallel_tensor_dim_idx_t> const &);

DimCoord<parallel_tensor_dim_idx_t> dim_coord_from_parallel_tensor_space_coord(
    ParallelTensorSpaceCoordinate const &);

} // namespace FlexFlow

#endif
