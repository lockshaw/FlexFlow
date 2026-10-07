#include "utils/benchmark_utils/random_int.h"

namespace FlexFlow {

int random_int(std::mt19937 &gen) {
  std::uniform_int_distribution<> dist;
  return dist(gen);
}

} // namespace FlexFlow
