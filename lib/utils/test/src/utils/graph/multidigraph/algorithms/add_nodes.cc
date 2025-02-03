#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/node/node_query.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("add_nodes(MultiDiGraph &, int)") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    std::unordered_set<Node> result = unordered_set_of(add_nodes(g, 3_n));
    std::unordered_set<Node> correct = g.query_nodes(node_query_all());

    CHECK(result == correct);
  }
