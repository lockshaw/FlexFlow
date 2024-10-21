#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_H

#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope_coordinate.dtg.h"
#include "utils/orthotope/orthotope_dim_idx_t.dtg.h"
#include <unordered_set>

namespace FlexFlow {

bool orthotope_contains_coord(Orthotope const &, OrthotopeCoordinate const &);

Orthotope restrict_orthotope_dims(Orthotope const &, std::unordered_set<orthotope_dim_idx_t> const &);

} // namespace FlexFlow

#endif
