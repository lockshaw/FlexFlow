#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_MAP_H

#include "utils/containers/contains_key.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <map>
#include <random>

namespace FlexFlow {

template <typename KF,
          typename VF,
          typename K = std::invoke_result_t<KF, std::mt19937 &>,
          typename V = std::invoke_result_t<VF, std::mt19937 &>>
std::map<K, V>
    random_map(std::mt19937 &gen, nonnegative_int size, KF &&kf, VF &&vf) {
  std::map<K, V> result;

  while (result.size() < size) {
    K k = kf(gen);
    if (!contains_key(result, k)) {
      V v = vf(gen);
      result.insert({k, v});
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
