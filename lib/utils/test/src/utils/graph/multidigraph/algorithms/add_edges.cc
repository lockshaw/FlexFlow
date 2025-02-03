#include "utils/graph/multidigraph/algorithms/add_edges.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/multidigraph/multidiedge_query.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("add_edges(MultiDiGraph &, std::vector<std::pair<Node, Node>>)") {
    MultiDiGraph g = MultiDiGraph::create<AdjacencyMultiDiGraph>();

    std::vector<Node> n = add_nodes(g, 3_n);

    std::vector<std::pair<Node, Node>> input = {
        {n.at(0), n.at(1)},
        {n.at(0), n.at(1)},
        {n.at(1), n.at(1)},
        {n.at(0), n.at(0)},
    };

    std::vector<MultiDiEdge> result = add_edges(g, input);

    auto src = [&](MultiDiEdge const &e) { return g.get_multidiedge_src(e); };

    auto dst = [&](MultiDiEdge const &e) { return g.get_multidiedge_dst(e); };

    SECTION("adds only those edges") {
      std::unordered_set<MultiDiEdge> added =
          g.query_edges(multidiedge_query_all());
      std::unordered_set<MultiDiEdge> returned = unordered_set_of(result);
      CHECK(returned == added);
    }

    SECTION("returns correct number of edges") {
      CHECK(result.size() == 4);
    }

    SECTION("returns unique edges") {
      CHECK(unordered_set_of(result).size() == result.size());
    }

    SECTION("edge 0") {
      CHECK(src(result.at(0)) == n.at(0));
      CHECK(dst(result.at(0)) == n.at(1));
    }

    SECTION("edge 1") {
      CHECK(src(result.at(1)) == n.at(0));
      CHECK(dst(result.at(1)) == n.at(1));
    }

    SECTION("edge 2") {
      CHECK(src(result.at(2)) == n.at(1));
      CHECK(dst(result.at(2)) == n.at(1));
    }

    SECTION("edge 3") {
      CHECK(src(result.at(3)) == n.at(0));
      CHECK(dst(result.at(3)) == n.at(0));
    }
  }
