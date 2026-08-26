#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_IS_LEFT_K_UNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_IS_LEFT_K_UNIQUE_H

#include "utils/binary_relation/binary_relation.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/require_all_same.h"
#include "utils/containers/values.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/require_all_same1.h"

namespace FlexFlow {

template <typename L, typename R>
std::optional<nonnegative_int> binary_relation_is_left_k_unique(BinaryRelation<L, R> const &rel)
{
  if (rel.empty()) {
    return 0_n;
  }

  std::map<R, nonnegative_int> preimage_sizes =
    generate_map(
      rel.right_values(),
      [&](R const &r) -> nonnegative_int {
        return num_elements(rel.at_r(r));
      });

  if (are_all_same(values(preimage_sizes))) {
    return require_all_same1(values(preimage_sizes));
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

#endif
