#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_AT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_AT_H

#include <optional>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
std::optional<V> try_at(std::unordered_map<K, V> const &m, K const &k) {
  auto it = m.find(k);
  if (it != m.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

#endif
