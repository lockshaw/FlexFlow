#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TAKE_WHILE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TAKE_WHILE_H

#include <vector>

namespace FlexFlow {

template <typename T, typename F>
std::vector<T> take_while(std::vector<T> const &v,
                          F &&f)
{
  std::vector<T> result;

  for (T const &t : v) {
    if (!f(t)) {
      break;
    }

    result.push_back(t);
  }

  return result;
}

} // namespace FlexFlow

#endif
