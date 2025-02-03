#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/add_edges.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_edges(MultiDiGraphView)") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    std::vector<Node> n = add_nodes(g, 3_n);
    std::vector<MultiDiEdge> e = add_edges(g,
                                           {
                                               {n.at(0), n.at(1)},
                                               {n.at(0), n.at(1)},
                                               {n.at(1), n.at(1)},
                                               {n.at(0), n.at(0)},
                                           });

    std::unordered_set<MultiDiEdge> result = get_edges(g);
    std::unordered_set<MultiDiEdge> correct = unordered_set_of(e);

    CHECK(result == correct);
  }
