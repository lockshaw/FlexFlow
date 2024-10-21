#include "utils/orthotope/orthotope_coordinate.h"
#include "utils/containers/filter_idxs.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/exception.h"
#include "utils/orthotope/orthotope_dim_idx_t.h"
#include "utils/fmt/set.h"

namespace FlexFlow {

std::set<orthotope_dim_idx_t> get_orthotope_coord_dims(OrthotopeCoordinate const &coord) {
  return dim_idxs_for_orthotope_with_num_dims(coord.idxs.size());
}

OrthotopeCoordinate restrict_orthotope_coord_dims(OrthotopeCoordinate const &coord, std::set<orthotope_dim_idx_t> const &mask) {
  std::set<orthotope_dim_idx_t> coord_dims = get_orthotope_coord_dims(coord);

  if (!is_subseteq_of(coord_dims, mask)) {
    throw mk_runtime_error(fmt::format("restrict_orthotope_coord_dims expected mask to be a subset of coord dims, but got coord={}, mask={}", coord, mask));
  }

  std::vector<int> new_idxs = filter_idxs(coord.idxs, [&](int i) { return contains(mask, orthotope_dim_idx_t{i}); });

  return OrthotopeCoordinate{new_idxs};
}

} // namespace FlexFlow
