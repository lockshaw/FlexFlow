#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/orthotope/dim_domain_biunique_mapping.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"

namespace FlexFlow {

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    empty_operator_space_to_ptensor_space_biunique_map()
{
  return OperatorSpaceToParallelTensorSpaceBiuniqueMapping{
      empty_dim_domain_biunique_mapping<operator_task_space_dim_idx_t,
        parallel_tensor_dim_idx_t>(),
  };
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    operator_ptensor_space_biunique_mapping_from_projection(
        DimProjection<operator_task_space_dim_idx_t,
                      parallel_tensor_dim_idx_t> const &projection,
        OperatorTaskSpace const &op_task_space,
        ParallelTensorDimDegrees const &parallel_tensor_dim_degrees)
{
  return OperatorSpaceToParallelTensorSpaceBiuniqueMapping{
    dim_domain_biunique_mapping_lift_right_domain(
      dim_domain_biunique_mapping_from_projection(
          /*projection=*/projection,
          /*l_domain=*/
          lift_minimal_dim_domain(
              minimal_dim_domain_from_operator_task_space(op_task_space)),
          /*r_domain=*/
          lift_minimal_dim_domain(
              minimal_dim_domain_from_parallel_tensor_dim_degrees(
                  parallel_tensor_dim_degrees)),
          /*l_dim_ordering=*/get_operator_task_space_dim_ordering(),
          /*r_dim_ordering=*/get_parallel_tensor_dim_ordering()),
      get_parallel_tensor_dim_indices(parallel_tensor_dim_degrees)),  
  };
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    operator_ptensor_space_biunique_mapping_from_composition(
        OperatorSpaceToParallelTensorSpaceBiuniqueMapping const &op_to_pt1_mapping,
        ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping const
            &pt1_to_pt2_mapping)
{
  return OperatorSpaceToParallelTensorSpaceBiuniqueMapping{
      compose_dim_domain_biunique_mappings(
          op_to_pt1_mapping.raw_mapping, pt1_to_pt2_mapping.raw_mapping),
  };
}

ParallelTensorDimDegrees get_parallel_tensor_space_for_biunique_mapping(
    OperatorSpaceToParallelTensorSpaceBiuniqueMapping const &mapping)
{
  return parallel_tensor_dim_degrees_from_dim_domain(
      mapping.raw_mapping.r_domain);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping get_identity_biunique_mapping(
    OperatorTaskSpace const &operator_task_space,
    ParallelTensorDimDegrees const &parallel_tensor_dim_degrees) 
{
  DimProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>
      projection = get_projection_for_op_to_ptensor_identity_mapping(
          operator_task_space_get_dim_idxs(operator_task_space), 
          get_nontrivial_parallel_tensor_dim_indices(parallel_tensor_dim_degrees));

  return operator_ptensor_space_biunique_mapping_from_projection(
      projection, operator_task_space, parallel_tensor_dim_degrees);
}

} // namespace FlexFlow
