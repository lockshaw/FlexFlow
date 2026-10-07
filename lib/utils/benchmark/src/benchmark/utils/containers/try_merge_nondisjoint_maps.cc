#include "benchmark/utils/containers/try_merge_nondisjoint_maps.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/benchmark_utils/random_hex_string.h"
#include "utils/benchmark_utils/random_int.h"
#include "utils/benchmark_utils/random_map.h"
#include "utils/containers/try_merge_nondisjoint_maps.h"

namespace FlexFlow {

void benchmark_try_merge_nondisjoint_maps(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  auto random_key = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 10_n);
  };

  auto random_value = [](std::mt19937 &gen) -> int { return random_int(gen); };

  std::map<std::string, int> lhs =
      random_map(gen, 50_n, random_key, random_value);
  std::map<std::string, int> rhs =
      random_map(gen, 50_n, random_key, random_value);

  LOOP(100, dry_run) {
    try_merge_nondisjoint_maps(lhs, rhs);
  }
}

} // namespace FlexFlow
