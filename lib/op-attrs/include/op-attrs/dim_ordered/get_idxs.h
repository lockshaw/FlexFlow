#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_GET_IDXS_H

#include "op-attrs/dim_ordered/dim_ordered.h"
#include "op-attrs/ff_dim_t.h"
#include "utils/containers/count.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

template <typename T>
std::vector<ff_dim_t> get_idxs(FFOrdered<T> const &d) {
  return transform(nonnegative_range(num_elements(d)),
                   [](nonnegative_int i) { return ff_dim_t{i}; });
}

} // namespace FlexFlow

#endif
