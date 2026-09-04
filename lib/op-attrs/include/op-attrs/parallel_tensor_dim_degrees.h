#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_DEGREES_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_DEGREES_H

#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

positive_int
    get_total_degree_of_ptensor_dim_degrees(ParallelTensorDimDegrees const &);

ParallelTensorDimDegrees
  mk_dim_degrees_from_ff_ordered(
    positive_int sum_degree,
    positive_int discard_copy_degree,
    std::vector<positive_int> const &shard_degrees);

num_ptensor_shard_dims_t
    get_ptensor_dim_degrees_num_shard_dims(ParallelTensorDimDegrees const &);
num_tensor_dims_t
    get_ptensor_dim_degrees_num_tensor_dims(ParallelTensorDimDegrees const &);

std::set<parallel_tensor_dim_idx_t>
    get_parallel_tensor_dim_indices(ParallelTensorDimDegrees const &);

std::set<parallel_tensor_dim_idx_t> get_nontrivial_parallel_tensor_dim_indices(
    ParallelTensorDimDegrees const &);

positive_int
    get_degree_for_parallel_tensor_dim_idx(ParallelTensorDimDegrees const &,
                                           parallel_tensor_dim_idx_t const &);

positive_int
    get_degree_for_relative_ff_dim_t(ParallelTensorDimDegrees const &,
                                     relative_ff_dim_t const &);

std::vector<positive_int>
    get_degrees_for_relative_ff_dim_interval(ParallelTensorDimDegrees const &degrees,
                                            relative_ff_dim_t const &start,
                                            relative_ff_dim_t const &end);

std::map<parallel_tensor_dim_idx_t, positive_int>
    get_parallel_tensor_degree_map(ParallelTensorDimDegrees const &);

ParallelTensorDimDegrees
    parallel_dim_degrees_drop_shard_dims(
      ParallelTensorDimDegrees const &,
      std::function<bool(ff_dim_t)> const &);

std::set<ParallelTensorSpaceCoordinate>
    get_parallel_tensor_space_coordinates(ParallelTensorDimDegrees const &);

ParallelTensorDimDegrees trivial_degrees_for_tensor_dims(TensorDims const &);

DimDomain<parallel_tensor_dim_idx_t>
    dim_domain_from_parallel_tensor_dim_degrees(
        ParallelTensorDimDegrees const &);

ParallelTensorDimDegrees parallel_tensor_dim_degrees_from_dim_domain(
    DimDomain<parallel_tensor_dim_idx_t> const &);

MinimalDimDomain<parallel_tensor_dim_idx_t>
    minimal_dim_domain_from_parallel_tensor_dim_degrees(
        ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
