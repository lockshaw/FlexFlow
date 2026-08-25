#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_IS_RIGHT_K_UNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_IS_RIGHT_K_UNIQUE_H

#include "utils/binary_relation/binary_relation.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/exception.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/values.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/are_all_same.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

template <typename L, typename R>
std::optional<nonnegative_int> binary_relation_is_right_k_unique(BinaryRelation<L, R> const &rel)
{
  if (rel.empty()) {
    return 0_n;
  }

  std::map<L, nonnegative_int> image_sizes = 
    generate_map(
      rel.left_values(),
      [&](L const &l) -> nonnegative_int {
        return num_elements(rel.at_l(l));
      });

  if (are_all_same(values(image_sizes))) {
    return require_all_same1(values(image_sizes));
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

#endif
