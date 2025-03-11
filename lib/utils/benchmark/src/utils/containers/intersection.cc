#include "utils/containers/intersection.h"
#include <benchmark/benchmark.h>
#include "../random_set.h"

using namespace ::FlexFlow;

static void benchmark_intersection(benchmark::State &state) {
  std::unordered_set<int> s1 = random_set(10000_n);
  std::unordered_set<int> s2 = random_set(10000_n);

  for (auto _ : state) {
    intersection(s1, s2);
  }
}

BENCHMARK(benchmark_intersection);
