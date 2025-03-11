#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONCEPTS_FMTABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONCEPTS_FMTABLE_H

#include "utils/check_fmtable.h"

namespace FlexFlow {

template <typename T>
concept Fmtable = is_fmtable_v<T>;

} // namespace FlexFlow

#endif
