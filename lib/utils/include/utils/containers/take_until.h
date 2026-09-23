#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TAKE_UNTIL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TAKE_UNTIL_H

#include <vector>

namespace FlexFlow {

template <typename T, typename F>
std::vector<T> take_until(std::vector<T> const &v,
                          F &&f)
{
  std::vector<T> result;

  for (T const &t : v) {
    if (f(t)) {
      break;
    }

    result.push_back(t);
  }

  return result;
}

} // namespace FlexFlow

#endif
