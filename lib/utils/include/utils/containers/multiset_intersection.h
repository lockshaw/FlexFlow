#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_INTERSECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_INTERSECTION_H

#include <set>
#include "utils/containers/set_union.h"
#include "utils/containers/set_of.h"
#include "utils/containers/range.h"
#include <optional>
#include <libassert/assert.hpp>
#include "utils/optional.h"

namespace FlexFlow {

template <typename T>
std::multiset<T> multiset_intersection(std::multiset<T> const &lhs,
                                       std::multiset<T> const &rhs)
{
  std::multiset<T> result;
  std::set<T> all_values = set_union(set_of(lhs), set_of(rhs));
  for (T const &t : all_values) {
    for (int i : range(std::min(lhs.count(t), rhs.count(t)))) {
      result.insert(t);
    }
  }

  return result;
}

template <typename C, typename T = typename C::value_type>
std::optional<T> multiset_intersection(C const &c) {
  std::optional<T> result;
  for (T const &t : c) {
    result = multiset_intersection(result.value_or(t), t);
  }

  return result;
}

template <typename C, typename T = typename C::value_type>
T multiset_intersection1(C const &c) {
  ASSERT(!c.empty());

  return assert_unwrap(multiset_intersection(c));
}


} // namespace FlexFlow

#endif
