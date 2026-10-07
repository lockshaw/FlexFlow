#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_SUBSET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_SUBSET_H

#include "utils/containers/is_subseteq_of.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/random_utils.h"
#include <libassert/assert.hpp>
#include <set>

namespace FlexFlow {

template <typename T>
std::set<T> random_subset(std::mt19937 &gen,
                          nonnegative_int size,
                          std::set<T> const &superset) {
  ASSERT(size <= superset.size());

  std::set<T> result;
  std::set<T> remaining = superset;

  while (result.size() < size) {
    T t = select_random(gen, remaining);
    result.insert(t);
    remaining.erase(t);
  }

  ASSERT(result.size() == size);
  ASSERT(is_subseteq_of(result, superset));

  return result;
}

} // namespace FlexFlow

#endif
