#ifndef _FLEXFLOW_LIB_UTILS_BENCHMARK_INCLUDE_INTERNAL_RANDOM_SET_H
#define _FLEXFLOW_LIB_UTILS_BENCHMARK_INCLUDE_INTERNAL_RANDOM_SET_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <random>

namespace FlexFlow {

template <typename F, typename T = std::invoke_result_t<F, std::mt19937 &>>
std::set<T> random_set(std::mt19937 &gen, nonnegative_int size, F &&f) {
  std::set<T> result;

  while (result.size() < size) {
    result.insert(f(gen));
  }

  return result;
}

} // namespace FlexFlow

#endif
