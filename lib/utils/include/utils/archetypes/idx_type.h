#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_IDX_TYPE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ARCHETYPES_IDX_TYPE_H

#include <cassert>
#include <string>

namespace FlexFlow {

template <int TAG>
struct idx_type {
  idx_type() = delete;  

  idx_type(idx_type const &) {
    assert(false);
  }

  idx_type &operator=(idx_type const &) {
    assert(false);
  }

  idx_type(idx_type &&) {
    assert(false);
  }

  idx_type &operator=(idx_type &&) {
    assert(false);
  }

  bool operator==(idx_type const &) const {
    assert(false);
  }

  bool operator!=(idx_type const &) const {
    assert(false);
  }
};

template <int TAG>
std::string format_as(idx_type<TAG> const &) {
  assert(false);
}

} // namespace FlexFlow

namespace std {

template <int TAG>
struct hash<::FlexFlow::idx_type<TAG>> {
  size_t operator()(::FlexFlow::idx_type<TAG> const &) const {
    assert(false);
  };
};

} // namespace std

#endif
