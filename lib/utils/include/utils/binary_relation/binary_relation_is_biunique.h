#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_IS_BIUNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_BINARY_RELATION_IS_BIUNIQUE_H

#include "utils/binary_relation/binary_relation.h"

namespace FlexFlow {

template <typename L, typename R>
bool binary_relation_is_biunique(BinaryRelation<L, R> const &rel)
{
  bool ls_are_unique = rel.left_value_occurences().size() == rel.left_values().size();
  bool rs_are_unique = rel.right_value_occurences().size() == rel.right_values().size();

  return ls_are_unique && rs_are_unique;
}

} // namespace FlexFlow

#endif
