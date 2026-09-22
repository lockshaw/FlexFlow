#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_MINUS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_MINUS_H

#include <set>
#include "utils/containers/range.h"

namespace FlexFlow {

template <typename T>
std::multiset<T> multiset_minus(std::multiset<T> const &lhs,
                                std::multiset<T> const &rhs)
{
  std::multiset<T> result;
  for (T const &t : lhs) {
    for (int i : range(lhs.count(t) - rhs.count(t))) {
      result.insert(t);
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
