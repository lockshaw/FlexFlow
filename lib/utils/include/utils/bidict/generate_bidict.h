#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_GENERATE_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_GENERATE_BIDICT_H

#include "utils/bidict/bidict.h"
#include "utils/containers/get_element_type.h"
#include "utils/containers/transform.h"
#include <type_traits>
#include "utils/concepts/hashable.h"

namespace FlexFlow {

template <typename F,
          typename C,
          Hashable K = get_element_type_t<C>,
          Hashable V = std::invoke_result_t<F, K>>
bidict<K, V> generate_bidict(C const &c, F const &f) {
  auto transformed = transform(c, [&](K const &k) -> std::pair<K, V> {
    return {k, f(k)};
  });
  return {transformed.cbegin(), transformed.cend()};
}

} // namespace FlexFlow

#endif
