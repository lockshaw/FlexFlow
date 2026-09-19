#include "utils/int_ge_two/algorithms/int_ge_two_range.h"

namespace FlexFlow {

std::set<int_ge_two> int_ge_two_range(int_ge_two max) {
  return
    transform(
      range(2, max.int_from_int_ge_two(),
      [&](int x) -> int_ge_two {
        return int_ge_two{x};
      }
}

} // namespace FlexFlow
