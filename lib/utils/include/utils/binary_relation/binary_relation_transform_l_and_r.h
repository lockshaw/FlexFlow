#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_TRANSFORM_L_AND_R_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_TRANSFORM_L_AND_R_H

#include "utils/binary_relation/binary_relation.h"

namespace FlexFlow {

template <typename L,
          typename R,
          typename FL,
          typename FR,
          typename L2 = typename std::invoke_result_t<FL, L>,
          typename R2 = typename std::invoke_result_t<FR, R>>
BinaryRelation<L2, R2> binary_relation_transform_l_and_r(
    BinaryRelation<L, R> const &rel, FL &&fl, FR &&fr) {
  BinaryRelation<L2, R2> result;

  for (std::pair<L, R> const &p : rel.unwrap_as_set()) {
    result.equate(fl(p.first), fr(p.second));
  }

  return result;
}

} // namespace FlexFlow

#endif
