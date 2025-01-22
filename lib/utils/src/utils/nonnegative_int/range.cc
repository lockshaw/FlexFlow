#include "utils/nonnegative_int/range.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<nonnegative_int> range(nonnegative_int start, nonnegative_int end, int step) {
  return transform(range(start.get_value(), end.get_value(), step), [](int x) { return nonnegative_int{x}; });
}

std::vector<nonnegative_int> range(nonnegative_int end) {
  return transform(range(end.get_value()), [](int x) { return nonnegative_int{x}; });
}

} // namespace FlexFlow
