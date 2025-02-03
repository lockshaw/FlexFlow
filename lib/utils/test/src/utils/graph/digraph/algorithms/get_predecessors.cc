#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_predecessors") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 6);

    add_edges(g,
              {
                  DirectedEdge{n.at(0), n.at(1)},
                  DirectedEdge{n.at(1), n.at(2)},
                  DirectedEdge{n.at(1), n.at(3)},
                  DirectedEdge{n.at(1), n.at(5)},
                  DirectedEdge{n.at(2), n.at(4)},
                  DirectedEdge{n.at(3), n.at(4)},
                  DirectedEdge{n.at(4), n.at(1)},
              });

    std::unordered_map<Node, std::unordered_set<Node>> correct = {
        {n.at(0), {}},
        {n.at(1), {n.at(0), n.at(4)}},
        {n.at(2), {n.at(1)}},
        {n.at(3), {n.at(1)}},
        {n.at(4), {n.at(2), n.at(3)}},
        {n.at(5), {n.at(1)}},
    };

    std::unordered_map<Node, std::unordered_set<Node>> result =
        get_predecessors(g);

    CHECK(result == correct);
  }
