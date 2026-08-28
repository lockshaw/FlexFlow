#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_TO_PARALLEL_TENSOR_SPACE_BIUNIQUE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_TO_PARALLEL_TENSOR_SPACE_BIUNIQUE_MAPPING_H

#include "utils/orthotope/dim_projection.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"

namespace FlexFlow {

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    parallel_tensor_space_biunique_mapping_from_projection(
        DimProjection<parallel_tensor_dim_idx_t,
                      parallel_tensor_dim_idx_t> const &projection,
        ParallelTensorDimDegrees const &l_degrees,
        ParallelTensorDimDegrees const &r_degrees);

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    parallel_tensor_space_biunique_mapping_from_coord_mapping(
        bidict<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate> const &coord_mapping,
        ParallelTensorDimDegrees const &l_degrees,
        ParallelTensorDimDegrees const &r_degrees);

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    invert_parallel_tensor_space_biunique_mapping(
        ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping const &);

} // namespace FlexFlow

#endif
