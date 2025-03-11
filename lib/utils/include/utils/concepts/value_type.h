#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONCEPTS_IS_VALUE_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONCEPTS_IS_VALUE_TYPE_H

#include <concepts>
#include "utils/concepts/fmtable.h"
#include "utils/concepts/hashable.h"

namespace FlexFlow {

template <typename T>
concept ValueType 
  = std::copyable<T> 
 && std::equality_comparable<T> 
 && Fmtable<T>
 && Hashable<T>;

} // namespace FlexFlow

#endif
