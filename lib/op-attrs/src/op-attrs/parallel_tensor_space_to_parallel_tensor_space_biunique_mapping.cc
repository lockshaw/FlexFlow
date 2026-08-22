#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_biunique_mapping.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/orthotope/dim_domain_biunique_mapping.h"

namespace FlexFlow {

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    parallel_tensor_space_biunique_mapping_from_projection(
        DimProjection<parallel_tensor_dim_idx_t,
                      parallel_tensor_dim_idx_t> const &projection,
        ParallelTensorDimDegrees const &l_degrees,
        ParallelTensorDimDegrees const &r_degrees) {

  return ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping{
      dim_domain_biunique_mapping_from_projection(
          /*projection=*/projection,
          /*l_domain=*/dim_domain_from_parallel_tensor_dim_degrees(l_degrees),
          /*r_domain=*/dim_domain_from_parallel_tensor_dim_degrees(r_degrees),
          /*l_dim_ordering=*/get_parallel_tensor_dim_ordering(),
          /*r_dim_ordering=*/get_parallel_tensor_dim_ordering()),
  };
}

ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping
    invert_parallel_tensor_space_biunique_mapping(
        ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping const &m) {
  return ParallelTensorSpaceToParallelTensorSpaceBiuniqueMapping{
      invert_dim_domain_biunique_mapping(m.raw_mapping),
  };
}

} // namespace FlexFlow
