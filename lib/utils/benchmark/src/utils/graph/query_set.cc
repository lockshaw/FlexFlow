#include "utils/graph/query_set.h"
#include <benchmark/benchmark.h>
#include "../random_set.h"

using namespace ::FlexFlow;

static void benchmark_query_set_apply(benchmark::State &state) {
  query_set<int> q = query_set<int>{random_set(10000_n)};
  std::unordered_set<int> input = random_set(10000_n);

  for (auto _ : state) {
    apply_query(q, input);
  }
}

BENCHMARK(benchmark_query_set_apply);
