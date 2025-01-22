#include "utils/orthotope/orthotope.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/filter_idxs.h"
#include "utils/containers/product.h"
#include "utils/containers/scanr.h"
#include "utils/containers/subvec.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip_strict.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/all_of.h"
#include "utils/containers/filter_idxs.h"
#include "utils/containers/contains.h"

namespace FlexFlow {

nonnegative_int get_orthotope_num_dims(Orthotope const &orthotope) {
  return num_elements(orthotope.dims);
}

nonnegative_int get_orthotope_volume(Orthotope const &orthotope) {
  return product(orthotope.dims);
}

std::unordered_set<OrthotopeCoord> get_all_coords_in_orthotope(Orthotope const &orthotope) {
  std::unordered_multiset<std::vector<nonnegative_int>> raw_coords = cartesian_product(transform(orthotope.dims, [](nonnegative_int dim_size) { return range(dim_size); }));
  
  return unordered_set_of(transform(raw_coords, [](std::vector<nonnegative_int> const &raw_coord) { return OrthotopeCoord{raw_coord}; }));
}

bool orthotope_contains_coord(Orthotope const &orthotope, OrthotopeCoord const &coord) {
  if (orthotope.dims.size() != coord.raw.size()) {
    throw mk_runtime_error(fmt::format("orthotope_contains_coord expected orthotope and coord to have the same number of dims, but received orthotope={}, coord={}", orthotope, coord));
  }

  return all_of(zip_strict(coord.raw, orthotope.dims), [](nonnegative_int c, nonnegative_int o) { return c < o; });
}

Orthotope restrict_orthotope_dims_to(Orthotope const &orthotope, std::set<nonnegative_int> const &allowed_dims) {
  return Orthotope{
    filter_idxs(orthotope.dims, [&](nonnegative_int idx) { return contains(allowed_dims, idx); }),
  };
}

nonnegative_int flatten_orthotope_coord(OrthotopeCoord const &coord, Orthotope const &orthotope) {
  if (orthotope.dims.size() != coord.raw.size()) {
    throw mk_runtime_error(fmt::format("flatten_orthotope_coord expected orthotope and coord to have the same number of dims, but received orthotope={}, coord={}", orthotope, coord));
  }

  std::vector<nonnegative_int> steps = scanr(orthotope.dims, nonnegative_int{1}, 
                                             [](nonnegative_int r, nonnegative_int accum) {
                                               return r * accum;
                                             });

  return product(zip_with_strict(coord.raw, subvec(steps, 0, -1), 
                                 [](nonnegative_int coord_val, nonnegative_int step) { return coord_val * step; }));

}

OrthotopeCoord unflatten_orthotope_coord(nonnegative_int flattened, Orthotope const &orthotope) {
  std::vector<nonnegative_int> steps = scanr(orthotope.dims, nonnegative_int{1}, 
                                             [](nonnegative_int r, nonnegative_int accum) {
                                               return r * accum;
                                             });

  return zip3_with_strict(orthotope.dims, 
                          subvec(steps, 0, -1), 
                          subvec(orthotope.dims, 1, 0), []() { TODO_COLIN_THIS_IS_WHAT_YOU_WERE_DOING });
}

} // namespace FlexFlow
