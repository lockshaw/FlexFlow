#include "utils/orthotope/orthotope.h"
#include "utils/containers/zip_with.h"
#include "utils/containers/all_of.h"
#include "utils/exception.h"

namespace FlexFlow {

bool orthotope_contains_coord(Orthotope const &o, OrthotopeCoordinate const &c) {
  if (o.dims.size() != c.idxs.size()) {
    throw mk_runtime_error(fmt::format("orthotope_contains_coord expected orthotope and coord to have the same number of dims, but received o={}, c={}", o, c));
  }

  return all_of(zip_with(o.dims, c.idxs, [](int dim_size, int dim_coord) { return dim_coord >= 0 && dim_coord < dim_size; }));
}

} // namespace FlexFlow
