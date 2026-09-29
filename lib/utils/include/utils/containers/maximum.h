#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H

#include <algorithm>
#include <fmt/format.h>
#include <libassert/assert.hpp>
#include <set>

namespace FlexFlow {

template <typename C>
typename C::value_type maximum(C const &c) {
  if (c.empty()) {
    PANIC(
        fmt::format("maximum expected non-empty container but received {}", c));
  }

  return *std::max_element(c.begin(), c.end());
}

template <typename T>
T maximum(std::set<T> const &ts) {
  if (ts.empty()) {
    PANIC(
        fmt::format("maximum expected non-empty container but received {}", ts));
  }

  auto it = ts.cend();
  it--;
  return *it;
}

} // namespace FlexFlow

#endif
