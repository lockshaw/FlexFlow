#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_TYPE_H

#include <optional>
#include "utils/remove_cvref.h"

namespace FlexFlow {

template <typename C, typename Enable = void>
struct get_element_type {
  using type = typename C::value_type;
};

template <typename T>
struct get_element_type<std::optional<T>> {
  using type = T;
};

template <typename T>
using get_element_type_t = typename get_element_type<remove_cvref_t<T>>::type;

} // namespace FlexFlow

#endif
