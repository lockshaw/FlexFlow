#include "utils/benchmark_utils/random_dag.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/random_utils.h"

namespace FlexFlow {

void random_dag(std::mt19937 &gen,
                DiGraph &g,
                nonnegative_int num_nodes,
                nonnegative_int num_edges) {
  nonnegative_int max_num_edges = [&] {
    int nn = num_nodes.int_from_nonnegative_int();

    return nonnegative_int{
        (nn * (nn - 1)) / 2,
    };
  }();

  ASSERT(num_edges < max_num_edges);

  std::vector<Node> n = add_nodes(g, num_nodes.unwrap_nonnegative());

  std::set<DirectedEdge> edges;
  while (edges.size() < num_edges) {
    Node n1 = select_random(gen, n);
    Node n2 = select_random(gen, n);

    if (n1 == n2) {
      continue;
    }

    Node src = std::min(n1, n2);
    Node dst = std::max(n1, n2);

    edges.insert(DirectedEdge{src, dst});
  }

  add_edges(g, vector_of(edges));
}

} // namespace FlexFlow
