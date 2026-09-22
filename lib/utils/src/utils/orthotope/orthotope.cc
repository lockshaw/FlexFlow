#include "utils/orthotope/orthotope.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/all_of.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter_idxs.h"
#include "utils/containers/product.h"
#include "utils/containers/scanr.h"
#include "utils/containers/set_of.h"
#include "utils/containers/slice.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip3_with_strict.h"
#include "utils/containers/zip_strict.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/range.h"
#include "utils/containers/maximum.h"
#include "utils/orthotope/orthotope_coord.h"
#include "utils/containers/require_all_same1.h"
#include "utils/fmt/set.h"
#include "utils/containers/tail.h"
#include "utils/int_ge_two/int_ge_two.h"
#include "utils/containers/require_same.h"
#include "utils/containers/prime_factorization.h"
#include "utils/containers/multiset_intersection.h"
#include "utils/containers/multiset_minus.h"

namespace FlexFlow {

Orthotope trivial_orthotope() {
  return Orthotope{
    /*dims=*/std::vector<positive_int>{},
  };
}


nonnegative_int orthotope_get_num_dims(Orthotope const &orthotope) {
  return num_elements(orthotope.dims);
}

positive_int orthotope_get_volume(Orthotope const &orthotope) {
  return product(orthotope.dims);
}

std::set<OrthotopeCoord>
    get_all_coords_in_orthotope(Orthotope const &orthotope) {
  std::multiset<std::vector<nonnegative_int>> raw_coords =
      cartesian_product(transform(orthotope.dims, [](positive_int dim_size) {
        return nonnegative_range(dim_size);
      }));

  return set_of(
      transform(raw_coords, [](std::vector<nonnegative_int> const &raw_coord) {
        return OrthotopeCoord{raw_coord};
      }));
}

bool orthotope_contains_coord(Orthotope const &orthotope,
                              OrthotopeCoord const &coord) {
  ASSERT(orthotope.dims.size() == coord.raw.size(),
         "orthotope_contains_coord expected orthotope and coord to have the "
         "same number of dims",
         orthotope,
         coord);

  return all_are_true(zip_with(
      coord.raw, orthotope.dims, [](nonnegative_int c, positive_int o) {
        return c < o;
      }));
}

Orthotope
    restrict_orthotope_to_dims(Orthotope const &orthotope,
                               std::set<nonnegative_int> const &allowed_dims) {
  return Orthotope{
      filter_idxs(
          orthotope.dims,
          [&](nonnegative_int idx) { return contains(allowed_dims, idx); }),
  };
}

nonnegative_int flatten_orthotope_coord(OrthotopeCoord const &coord,
                                        Orthotope const &orthotope) {
  ASSERT(orthotope.dims.size() == coord.raw.size(),
         "flatten_orthotope_coord expected orthotope and coord to have the "
         "same number of dims",
         orthotope,
         coord);
  ASSERT(orthotope_contains_coord(orthotope, coord));

  std::vector<positive_int> steps =
      scanr(orthotope.dims, 1_p, [](positive_int r, positive_int accum) {
        return r * accum;
      });

  nonnegative_int result =
      sum(zip_with_strict(coord.raw,
                          slice(steps, 1, std::nullopt),
                          [](nonnegative_int coord_val, positive_int step) {
                            return coord_val * step;
                          }));

  ASSERT(result <= orthotope_get_maximum_offset(orthotope));

  return result;
}

OrthotopeCoord orthotope_get_maximum_coord(Orthotope const &orthotope) {
  return OrthotopeCoord{
      transform(orthotope.dims,
                [](positive_int d) {
                  return nonnegative_int{d.int_from_positive_int() - 1};
                }),
  };
}

nonnegative_int orthotope_get_maximum_offset(Orthotope const &orthotope) {
  return nonnegative_int{product(orthotope.dims).int_from_positive_int() - 1};
}

OrthotopeCoord unflatten_orthotope_coord(nonnegative_int flattened,
                                         Orthotope const &orthotope) {
  ASSERT(flattened <= orthotope_get_maximum_offset(orthotope));

  std::vector<positive_int> steps =
      scanr(orthotope.dims, 1_p, [](positive_int r, positive_int accum) {
        return r * accum;
      });

  OrthotopeCoord result = OrthotopeCoord{
      zip3_with_strict(
          orthotope.dims,
          slice(steps, 1, std::nullopt),
          slice(steps, 0, -1),
          [&](positive_int dim, positive_int step, positive_int next_step) {
            return (flattened % next_step) / step;
          }),
  };

  ASSERT(orthotope_contains_coord(orthotope, result));

  return result;
}

Orthotope smallest_orthotope_for_coord_set(std::set<OrthotopeCoord> const &coord_set)
{
  if (coord_set.size() == 0) {
    return trivial_orthotope();
  }

  nonnegative_int num_dims =
    require_all_same1(
      transform(coord_set,
                [](OrthotopeCoord const &c) -> nonnegative_int
                {
                  return orthotope_coord_num_dims(c);
                }));

  auto component_for_idx = [&](nonnegative_int idx) -> positive_int {
    return maximum(
      transform(coord_set,
                [&](OrthotopeCoord const &c) -> positive_int {
                  return c.raw.at(idx.int_from_nonnegative_int()) + 1_p;
                }));
  };

  return Orthotope{
    transform(
      nonnegative_range(0_n, num_dims),
      component_for_idx),
  };
}

std::optional<Orthotope> strict_orthotope_for_coord_set(std::set<OrthotopeCoord> const &coord_set) {
  Orthotope smallest = smallest_orthotope_for_coord_set(coord_set);

  if (get_all_coords_in_orthotope(smallest) == coord_set) {
    return smallest;
  } else {
    return std::nullopt;
  }
}

std::optional<Orthotope> orthotope_tail(Orthotope const &o) {
  if (orthotope_get_num_dims(o) == 0) {
    return std::nullopt;
  } else {
    return Orthotope{
      /*dims=*/tail(o.dims),
    };
  }
}

bool is_orthotope_divisor_of(Orthotope const &dividend,
                             Orthotope const &divisor) 
{
  nonnegative_int dividend_num_dims = orthotope_get_num_dims(dividend);
  nonnegative_int divisor_num_dims = orthotope_get_num_dims(divisor);

  if (dividend_num_dims != divisor_num_dims) {
    return false;
  }

  nonnegative_int num_dims = 
    require_same(dividend_num_dims, divisor_num_dims);

  return all_of(  
    nonnegative_range(num_dims),
    [&](nonnegative_int dim_idx) -> bool {
      int d = dim_idx.int_from_nonnegative_int();
      return dividend.dims.at(d) % divisor.dims.at(d) == 0;
    });
}

Orthotope orthotope_find_lexicographically_first_divisor_of_volume(
    Orthotope const &dividend,
    positive_int divisor_volume) 
{
  ASSERT(divisor_volume <= orthotope_get_volume(dividend));

  std::vector<std::multiset<int_ge_two>> dividend_dim_factorizations = 
    transform(dividend.dims,
              [](positive_int d) -> std::multiset<int_ge_two> {
                return prime_factorization(d);
              });

  std::multiset<int_ge_two> divisor_volume_factorization = prime_factorization(divisor_volume);

  std::multiset<int_ge_two> remaining_divisor_volume_factors = divisor_volume_factorization;
  std::vector<std::multiset<int_ge_two>> divisor_dim_factorizations;

  for (std::multiset<int_ge_two> const &dividend_dim_factorization : dividend_dim_factorizations) {
    std::multiset<int_ge_two> divisor_dim_factorization = multiset_intersection(
      remaining_divisor_volume_factors,
      dividend_dim_factorization);

    remaining_divisor_volume_factors = 
      multiset_minus(remaining_divisor_volume_factors, divisor_dim_factorization);

    divisor_dim_factorizations.push_back(divisor_dim_factorization);
  }
  ASSERT(remaining_divisor_volume_factors.size() == 0);

  auto as_positive_ints = [&](std::multiset<int_ge_two> const &xs) 
    -> std::multiset<positive_int>
  {
    return transform(xs,
                     [](int_ge_two x) -> positive_int {
                       return x.positive_int_from_int_ge_two(); 
                     });
  };

  std::vector<positive_int> divisor_dims = 
    transform(
      divisor_dim_factorizations,
      [&](std::multiset<int_ge_two> const &divisor_dim_factorization) -> positive_int {
        return product(as_positive_ints(divisor_dim_factorization));
      });

  Orthotope divisor = Orthotope{
    /*dims=*/divisor_dims,
  };

  ASSERT(is_orthotope_divisor_of(dividend, divisor));
  ASSERT(orthotope_get_volume(divisor) == divisor_volume);

  return divisor;
}


} // namespace FlexFlow
