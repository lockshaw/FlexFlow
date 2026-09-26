#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_WITH_IDX_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_WITH_IDX_H

#include <vector>
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

template <typename In, 
          typename F,
          typename Out = std::invoke_result_t<F, nonnegative_int, In>>
std::vector<Out> transform_with_idx(std::vector<In> const &v, F &&f)
{
  std::vector<Out> result;
  for (nonnegative_int idx : nonnegative_range(num_elements(v))) {
    result.push_back(f(idx, v.at(idx.int_from_nonnegative_int())));
  }
  return result;
}

} // namespace FlexFlow

#endif
