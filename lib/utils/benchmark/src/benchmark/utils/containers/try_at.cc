#include "benchmark/utils/containers/try_at.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/benchmark_utils/random_hex_string.h"
#include "utils/benchmark_utils/random_int.h"
#include "utils/benchmark_utils/random_map.h"
#include "utils/benchmark_utils/random_unordered_map.h"
#include "utils/benchmark_utils/random_vector.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

void benchmark_try_at_for_map(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  auto random_key = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 3_n);
  };

  std::map<std::string, int> m =
      random_map(gen, 1000_n, random_key, random_int);

  std::vector<std::string> queries = random_vector(gen, 800_n, random_key);

  LOOP(1, dry_run) {
    for (std::string const &q : queries) {
      try_at(m, q);
    }
  }
}

void benchmark_try_at_for_unordered_map(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  auto random_key = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 3_n);
  };

  std::unordered_map<std::string, int> m =
      random_unordered_map(gen, 1000_n, random_key, random_int);

  std::vector<std::string> queries = random_vector(gen, 800_n, random_key);

  LOOP(1, dry_run) {
    for (std::string const &q : queries) {
      try_at(m, q);
    }
  }
}

} // namespace FlexFlow
