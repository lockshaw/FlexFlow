#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_TRY_AT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_TRY_AT_H

#include "op-attrs/dim_ordered/dim_ordered.h"

namespace FlexFlow {

template <typename Idx, typename T>
std::optional<T> try_at(DimOrdered<Idx, T> const &c, Idx const &idx) {
  if (c.idx_is_valid(idx)) {
    return c.at(idx);
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

#endif
