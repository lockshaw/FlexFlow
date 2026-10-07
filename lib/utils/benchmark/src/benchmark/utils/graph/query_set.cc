#include "utils/graph/query_set.h"
#include "benchmark/utils/graph/query_set.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/benchmark_utils/random_int.h"
#include "utils/benchmark_utils/random_set.h"
#include "utils/benchmark_utils/random_vector.h"

namespace FlexFlow {

void benchmark_query_set_allowed_values(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  std::set<int> s = random_set(gen, 1000_n, random_int);
  query_set<int> q = query_set<int>::match_values_in(s);

  LOOP(1000, dry_run) {
    allowed_values(q);
  }
}

void benchmark_query_set_apply_query_to_set(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  std::set<int> s1 = random_set(gen, 100_n, random_int);
  query_set<int> q = query_set<int>::match_values_in(s1);

  std::set<int> s2 = random_set(gen, 100_n, random_int);

  LOOP(1000, dry_run) {
    apply_query(q, s2);
  }
}

void benchmark_query_set_apply_query_to_vector(bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  std::set<int> s = random_set(gen, 100_n, random_int);
  query_set<int> q = query_set<int>::match_values_in(s);

  std::vector<int> vec = random_vector(gen, 100_n, random_int);

  LOOP(1000, dry_run) {
    apply_query(q, vec);
  }
}

} // namespace FlexFlow
