#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_IDX_TYPE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_IDX_TYPE_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/concepts/value_type.h"

namespace FlexFlow {

template <ValueType T>
struct IdxType {
  static T wrap_idx(nonnegative_int const &) = delete;
  static nonnegative_int unwrap_idx(T const &) = delete;
};

} // namespace FlexFlow

#endif
