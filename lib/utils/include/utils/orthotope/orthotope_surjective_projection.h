#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_SURJECTIVE_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_SURJECTIVE_PROJECTION_H

#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope_surjective_projection.dtg.h"
#include "utils/orthotope/orthotope_coordinate.dtg.h"
#include <unordered_set>

namespace FlexFlow {

OrthotopeSurjectiveProjection
  make_orthotope_projection_from_map(std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> const &);

std::unordered_map<orthotope_dim_idx_t, orthotope_dim_idx_t> get_src_to_dst_dim_map(OrthotopeSurjectiveProjection const &);

orthotope_dim_idx_t get_dst_dim_for_src_dim(OrthotopeSurjectiveProjection const &, orthotope_dim_idx_t const &);

int get_src_num_dims(OrthotopeSurjectiveProjection const &);
int get_dst_num_dims(OrthotopeSurjectiveProjection const &);

OrthotopeSurjectiveProjection reverse_projection(OrthotopeSurjectiveProjection const &);

std::unordered_set<OrthotopeSurjectiveProjection> get_all_surjective_projections_between(Orthotope const &src, Orthotope const &dst);

int deconflict_overlapping_dims(std::vector<std::pair<int, int>> const &coords_and_sizes);
OrthotopeCoordinate project_coordinate_through(OrthotopeSurjectiveProjection const &, Orthotope const &, OrthotopeCoordinate const &);

} // namespace FlexFlow

#endif
