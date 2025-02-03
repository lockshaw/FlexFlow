#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("is_acyclic") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 6);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
                  DirectedEdge{n.at(1), n.at(5)},
                  DirectedEdge{n.at(2), n.at(4)},
                  DirectedEdge{n.at(3), n.at(1)},
                  DirectedEdge{n.at(3), n.at(4)},
              });

    std::optional<bool> correct = false;
    std::optional<bool> result = is_acyclic(g);

    CHECK(result == correct);
  }
