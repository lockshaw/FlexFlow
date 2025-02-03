#include "utils/random_utils.h"

namespace FlexFlow {

float randf() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

} // namespace FlexFlow
