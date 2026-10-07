#include "utils/graph/digraph/algorithms/transitive_closure.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/benchmark_utils/random_dag.h"
#include "utils/graph/instances/adjacency_digraph.h"

namespace FlexFlow {

void benchmark_transitive_closure(std::mt19937 &gen, bool dry_run) {
  nonnegative_int num_nodes = 50_n;
  nonnegative_int num_edges = 200_n;

  DiGraph g = DiGraph::create<AdjacencyDiGraph>();

  random_dag(gen, g, num_nodes, num_edges);

  LOOP(50, dry_run) {
    transitive_closure(g);
  }
}

} // namespace FlexFlow
