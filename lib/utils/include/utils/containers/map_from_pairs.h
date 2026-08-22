#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_PAIRS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_PAIRS_H

#include <map>
#include "utils/containers/contains_key.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename C,
          typename K = typename C::value_type::first_type,
          typename V = typename C::value_type::second_type>
std::map<K, V> map_from_pairs(C const &c) {
  std::map<K, V> result;

  for (std::pair<K, V> const &kv : c) {
    ASSERT(!contains_key(result, kv.first));

    result.insert(kv);
  }

  return result;
}

} // namespace FlexFlow

#endif
