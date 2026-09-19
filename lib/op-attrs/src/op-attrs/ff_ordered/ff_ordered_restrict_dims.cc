#include "op-attrs/ff_ordered/ff_ordered_restrict_dims.h"

namespace FlexFlow {

using T = value_type<0>;

template
  FFOrdered<T> ff_ordered_restrict_dims(FFOrdered<T> const &,
                                        std::set<ff_dim_t> const &);

} // namespace FlexFlow
