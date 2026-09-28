#include "benchmark/utils/graph/digraph/algorithms/transitive_closure.h"
#include "benchmark/utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/benchmark_utils.h"
#include <random>
#include <string>
#include <vector>

using namespace ::FlexFlow;

int main(int argc, char **argv) {
  std::map<std::string, std::function<void(bool)>> benchmarks = {
      {
          "transitive_closure",
          [](bool dry_run) -> void {
            std::mt19937 gen;
            gen.seed(0);
            benchmark_transitive_closure(
                /*gen=*/gen,
                /*edge_percentage=*/0.5,
                /*num_nodes=*/100,
                /*dry_run=*/dry_run);
          },
      },
      {
          "transitive_reduction",
          [](bool dry_run) -> void {
            std::mt19937 gen;
            gen.seed(0);
            benchmark_transitive_reduction(
                /*gen=*/gen,
                /*edge_percentage=*/0.5,
                /*num_nodes=*/100,
                /*dry_run=*/dry_run);
          },
      },
  };

  benchmark_main(argc, argv, benchmarks);
}
