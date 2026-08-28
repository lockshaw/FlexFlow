#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_BIUNIQUE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_BIUNIQUE_MAPPING_H

#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "utils/orthotope/dim_projection.dtg.h"

namespace FlexFlow {

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    empty_operator_space_to_ptensor_space_biunique_map();

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    operator_ptensor_space_biunique_mapping_from_projection(
        DimProjection<operator_task_space_dim_idx_t,
                      parallel_tensor_dim_idx_t> const &projection,
        OperatorTaskSpace const &op_task_space,
        ParallelTensorDimDegrees const &parallel_tensor_dim_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    operator_ptensor_space_biunique_mapping_from_composition(
        OperatorSpaceToParallelTensorSpaceBiuniqueMapping const &op_to_pt1_mapping,
        ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping const
            &pt1_to_pt2_mapping);

ParallelTensorDimDegrees get_parallel_tensor_space_for_biunique_mapping(
    OperatorSpaceToParallelTensorSpaceBiuniqueMapping const &mapping);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping get_identity_biunique_mapping(
    OperatorTaskSpace const &operator_task_space,
    ParallelTensorDimDegrees const &parallel_tensor_dim_degrees);

} // namespace FlexFlow

#endif
