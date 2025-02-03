#include "utils/graph/digraph/algorithms/transitive_closure.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <catch2/catch_test_macros.hpp>
#include "utils/containers/vector_of.h"
#include "utils/graph/algorithms.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/random_utils.h"
#include <catch2/benchmark/catch_benchmark.hpp>

using namespace ::FlexFlow;

DiGraphView random_dag(nonnegative_int num_nodes,
                       float edges_fraction) {
  assert (edges_fraction <= 1.0);
  assert (edges_fraction >= 0.0);

  int max_num_edges = [&] {
    int nn = num_nodes.unwrap_nonnegative();

    return (nn * (nn - 1)) / 2;
  }(); 

  nonnegative_int num_edges = nonnegative_int{
    static_cast<int>(max_num_edges * edges_fraction),
  };

  assert (num_edges <= max_num_edges);

  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, num_nodes.unwrap_nonnegative());

  std::unordered_set<DirectedEdge> edges;
  while (edges.size() < num_edges) {
    Node n1 = select_random(n);
    Node n2 = select_random(n);

    if (n1 == n2) { 
      continue;
    }

    Node src = std::min(n1, n2);
    Node dst = std::max(n1, n2);

    edges.insert(DirectedEdge{src, dst});
  }

  add_edges(g, vector_of(edges));

  return g;
}

  TEST_CASE("transitive_closure(DiGraphView)") {
    {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();

      SECTION("maximum number of new edges") {
        std::vector<Node> n = add_nodes(g, 5);

        add_edges(g,
                  {
                      DirectedEdge{n.at(0), n.at(1)},
                      DirectedEdge{n.at(1), n.at(2)},
                      DirectedEdge{n.at(2), n.at(3)},
                      DirectedEdge{n.at(3), n.at(4)},
                  });

        DiGraphView result = transitive_closure(g);

        SECTION("nodes") {
          std::unordered_set<Node> result_nodes = get_nodes(result);
          std::unordered_set<Node> correct_nodes = unordered_set_of(n);
          CHECK(result_nodes == correct_nodes);
        }

        SECTION("edges") {
          std::unordered_set<DirectedEdge> result_edges = get_edges(result);
          std::unordered_set<DirectedEdge> correct_edges = {
              DirectedEdge{n.at(0), n.at(1)},
              DirectedEdge{n.at(0), n.at(2)},
              DirectedEdge{n.at(0), n.at(3)},
              DirectedEdge{n.at(0), n.at(4)},
              DirectedEdge{n.at(1), n.at(2)},
              DirectedEdge{n.at(1), n.at(3)},
              DirectedEdge{n.at(1), n.at(4)},
              DirectedEdge{n.at(2), n.at(3)},
              DirectedEdge{n.at(2), n.at(4)},
              DirectedEdge{n.at(3), n.at(4)},
          };
          CHECK(result_edges == correct_edges);
        }
      }
    }

    for (float edge_fraction : std::vector{0.25, 0.5, 0.75}) {
      DYNAMIC_SECTION("transitive_reduction(100, " << edge_fraction << ")") {
        DiGraphView g = random_dag(100_n, edge_fraction);
        BENCHMARK("transitive_reduction") {
          return transitive_closure(g);
        };
      }
    }
  }
