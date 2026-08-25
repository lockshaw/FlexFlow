#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_REQUIRE_BINARY_RELATION_IS_BIUNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BINARY_RELATION_REQUIRE_BINARY_RELATION_IS_BIUNIQUE_H

#include "utils/bidict/bidict.h"
#include "utils/binary_relation/binary_relation.h"
namespace FlexFlow {

template <typename L, typename R>
bidict<L, R> require_binary_relation_is_biunique(BinaryRelation<L, R> const &rel)
{
  bidict<L, R> result;
  for (auto const &[l, r] : rel.unwrap_as_set()) {
    result.equate(l, r);
  }

  return result;
}

} // namespace FlexFlow

#endif
