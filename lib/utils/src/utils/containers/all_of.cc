#include "utils/containers/all_of.h"

namespace FlexFlow {

bool all_of(std::vector<bool> const &v) {
  for (bool v : v) {
    if (!v) {
      return false;
    }
  }

  return true;
}

} // namespace FlexFlow
