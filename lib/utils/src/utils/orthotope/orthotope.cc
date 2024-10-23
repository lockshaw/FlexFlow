#include "utils/orthotope/orthotope.h"
#include "utils/containers/product.h"
#include "utils/orthotope/orthotope_dim_indexed/drop_idxs_except.h"
#include "utils/orthotope/orthotope_dim_indexed/zip_with.h"
#include "utils/orthotope/orthotope_dim_indexed/all_of.h"
#include "utils/containers/all_of.h"
#include "utils/exception.h"

namespace FlexFlow {

std::set<orthotope_dim_idx_t> get_orthotope_dims(Orthotope const &orthotope) {
  return orthotope.dims.indices();
}

bool orthotope_contains_coord(Orthotope const &o, OrthotopeCoordinate const &c) {
  if (o.dims.size() != c.idxs.size()) {
    throw mk_runtime_error(fmt::format("orthotope_contains_coord expected orthotope and coord to have the same number of dims, but received o={}, c={}", o, c));
  }

  return all_of(zip_with(o.dims, c.idxs, [](int dim_size, int dim_coord) { return dim_coord >= 0 && dim_coord < dim_size; }));
}

int orthotope_get_volume(Orthotope const &o) {
  return product(o.dims.get_contents());
}

Orthotope orthotope_drop_dims_except(Orthotope const &o, std::set<orthotope_dim_idx_t> const &keep) {
  return Orthotope{drop_idxs_except(o.dims, keep)};
}

} // namespace FlexFlow
