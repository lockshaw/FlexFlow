#include "benchmark/utils/containers/transform.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/benchmark_utils/random_int.h"
#include "utils/benchmark_utils/random_set.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

void benchmark_transform_set(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  std::set<int> s = random_set(gen, 500_n, random_int);

  auto f = [&](int x) -> std::pair<int, int> {
    return std::pair{x + 1, x + 2};
  };

  LOOP(30, dry_run) {
    transform(s, f);
  }
}

} // namespace FlexFlow
