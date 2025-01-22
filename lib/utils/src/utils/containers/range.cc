#include "utils/containers/range.h"
#include <cassert>
#include <fmt/format.h>
#include "utils/exception.h"

namespace FlexFlow {

std::vector<int> range(int start, int end, int step) {
  if (step == 0) {
    throw mk_runtime_error(fmt::format("range expected step != 0, but received: {}", step));
  }

  std::vector<int> result;
  if (step > 0) {
    for (int i = start; i < end; i += step) {
      result.push_back(i);
    }
  } else {
    for (int i = start; i > end; i += step) {
      result.push_back(i);
    }
  }
  return result;
}

std::vector<int> range(int end) {
  return range(0, end);
}

} // namespace FlexFlow
