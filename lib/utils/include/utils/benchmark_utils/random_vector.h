#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_VECTOR_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <random>

namespace FlexFlow {

template <typename F, typename T = std::invoke_result_t<F, std::mt19937 &>>
std::vector<T> random_vector(std::mt19937 &gen, nonnegative_int size, F &&f) {
  std::vector<T> result;

  while (result.size() < size) {
    result.push_back(f(gen));
  }

  return result;
}

} // namespace FlexFlow

#endif
