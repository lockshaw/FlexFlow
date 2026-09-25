#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_MINUS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_MINUS_H

#include <set>
#include "utils/containers/range.h"
#include "utils/integer_conversions.h"
#include "utils/containers/set_of.h"

namespace FlexFlow {

template <typename T>
std::multiset<T> multiset_minus(std::multiset<T> const &lhs,
                                std::multiset<T> const &rhs)
{
  std::multiset<T> result;
  for (T const &t : set_of(lhs)) {
    int result_count = std::max(
      0,
      int_from_size_t(lhs.count(t)) - int_from_size_t(rhs.count(t))
    );
    for (int i : range(result_count)) {
      result.insert(t);
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
