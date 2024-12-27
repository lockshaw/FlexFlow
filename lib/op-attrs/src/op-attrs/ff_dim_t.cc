#include "op-attrs/ff_dim_t.h"
#include "utils/containers/range.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"

using ::FlexFlow::ff_dim_t;

namespace FlexFlow {

std::set<ff_dim_t> ff_dim_range(int end) {
  return set_of(transform(range(end), [](int i) { return ff_dim_t{i}; }));
}

} // namespace FlexFlow

namespace rc {

Gen<ff_dim_t> Arbitrary<ff_dim_t>::arbitrary() {
  return gen::construct<ff_dim_t>(
      gen::inRange<int>(0, MAX_TENSOR_DIM));
}
  

} // namespace rc
