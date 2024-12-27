#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_DIM_T_H

#include "op-attrs/ff_dim.dtg.h"
#include "rapidcheck.h"

namespace FlexFlow {

std::set<ff_dim_t> ff_dim_range(int end);

} // namespace FlexFlow

namespace rc {
template <>
struct Arbitrary<FlexFlow::ff_dim_t> {
  static Gen<FlexFlow::ff_dim_t> arbitrary();
};
} // namespace rc

#endif 
