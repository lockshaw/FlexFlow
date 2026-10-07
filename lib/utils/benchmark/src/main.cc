#include "benchmark/utils/containers/contains.h"
#include "benchmark/utils/containers/transform.h"
#include "benchmark/utils/graph/digraph/algorithms/transitive_closure.h"
#include "benchmark/utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/benchmark_utils/benchmark_main.h"
#include <random>
#include <string>
#include <vector>

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  std::map<std::string, std::function<void(bool)>> benchmarks = {
      {
          "contains_for_set",
          benchmark_contains_for_set,
      },
      {
          "transform_set",
          benchmark_transform_set,
      },
      {
          "transitive_closure",
          benchmark_transitive_closure,
      },
      {
          "transitive_reduction",
          benchmark_transitive_reduction,
      },
  };

  benchmark_main(argc, argv, benchmarks);
}
