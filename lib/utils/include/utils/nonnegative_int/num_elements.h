#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NUM_ELEMENTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NUM_ELEMENTS_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <vector>
#include "utils/integer_conversions.h"

namespace FlexFlow {

template <typename T>
nonnegative_int num_elements(std::vector<T> const &v) {
  return nonnegative_int{int_from_size_t(v.size())};
}

template <typename T>
nonnegative_int num_elements(std::list<T> const &v) {
  return nonnegative_int{int_from_size_t(v.size())};
}

template <typename T>
nonnegative_int num_elements(std::set<T> const &v) {
  return nonnegative_int{int_from_size_t(v.size())};
}

template <typename T>
nonnegative_int num_elements(std::unordered_set<T> const &v) {
  return nonnegative_int{int_from_size_t(v.size())};
}

} // namespace FlexFlow

#endif
