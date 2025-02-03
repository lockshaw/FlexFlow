#include "utils/graph/digraph/algorithms/get_edges_from_subgraph_to_subgraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_edges_from_subgraph_to_subgraph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 5);
    SECTION("basic tests") {
      std::unordered_set<Node> src_subgraph = {n.at(0), n.at(1), n.at(4)};
      std::unordered_set<Node> dst_subgraph = {n.at(2), n.at(3)};

      SECTION("returns all edges between subgraphs") {
        std::vector<DirectedEdge> e = {
            DirectedEdge{n.at(0), n.at(2)},
            DirectedEdge{n.at(0), n.at(3)},
            DirectedEdge{n.at(1), n.at(3)},
            DirectedEdge{n.at(4), n.at(2)},
        };

        add_edges(g, e);

        std::unordered_set<DirectedEdge> result =
            get_edges_from_subgraph_to_subgraph(g, src_subgraph, dst_subgraph);
        std::unordered_set<DirectedEdge> correct = unordered_set_of(e);

        CHECK(result == correct);
      }

      SECTION("does not return reverse edges") {
        std::vector<DirectedEdge> e = {
            DirectedEdge{n.at(1), n.at(3)},
            DirectedEdge{n.at(2), n.at(0)},
        };

        add_edges(g, e);

        std::unordered_set<DirectedEdge> result =
            get_edges_from_subgraph_to_subgraph(g, src_subgraph, dst_subgraph);
        std::unordered_set<DirectedEdge> correct = {e.at(0)};

        CHECK(result == correct);
      }

      SECTION("does not return edges within subgraph") {
        std::vector<DirectedEdge> e = {
            DirectedEdge{n.at(0), n.at(1)},
            DirectedEdge{n.at(0), n.at(3)},
        };

        add_edges(g, e);

        std::unordered_set<DirectedEdge> result =
            get_edges_from_subgraph_to_subgraph(g, src_subgraph, dst_subgraph);
        std::unordered_set<DirectedEdge> correct = {e.at(1)};

        CHECK(result == correct);
      }

      SECTION("returns no edges if there are no edges from src_subgraph to "
              "dst_subgraph") {
        std::vector<DirectedEdge> e = {
            DirectedEdge{n.at(0), n.at(1)},
            DirectedEdge{n.at(2), n.at(3)},
        };

        add_edges(g, e);

        std::unordered_set<DirectedEdge> result =
            get_edges_from_subgraph_to_subgraph(g, src_subgraph, dst_subgraph);
        std::unordered_set<DirectedEdge> correct = {};

        CHECK(result == correct);
      }
    }

    SECTION("empty subgraphs") {
      std::vector<DirectedEdge> e = {
          DirectedEdge{n.at(0), n.at(2)},
          DirectedEdge{n.at(0), n.at(3)},
          DirectedEdge{n.at(1), n.at(3)},
      };

      add_edges(g, e);

      SECTION("returns no edges if no nodes in src_subgraph") {
        std::unordered_set<DirectedEdge> result =
            get_edges_from_subgraph_to_subgraph(g, {}, unordered_set_of(n));
        std::unordered_set<DirectedEdge> correct = {};

        CHECK(result == correct);
      }

      SECTION("returns no edges if no nodes in dst_subgraph") {
        std::unordered_set<DirectedEdge> result =
            get_edges_from_subgraph_to_subgraph(g, unordered_set_of(n), {});
        std::unordered_set<DirectedEdge> correct = {};

        CHECK(result == correct);
      }

      SECTION("returns no edges if both subgraphs are empty") {
        std::unordered_set<DirectedEdge> result =
            get_edges_from_subgraph_to_subgraph(g, {}, {});
        std::unordered_set<DirectedEdge> correct = {};

        CHECK(result == correct);
      }
    }

    SECTION("if subgraphs do not cover graph, then does not return external "
            "edges") {
      std::vector<DirectedEdge> e = {
          DirectedEdge{n.at(0), n.at(2)},
          DirectedEdge{n.at(0), n.at(3)},
          DirectedEdge{n.at(1), n.at(3)},
      };

      add_edges(g, e);

      std::unordered_set<Node> src_subgraph = {n.at(0)};
      std::unordered_set<Node> dst_subgraph = {n.at(3)};

      std::unordered_set<DirectedEdge> result =
          get_edges_from_subgraph_to_subgraph(g, src_subgraph, dst_subgraph);
      std::unordered_set<DirectedEdge> correct = {e.at(1)};

      CHECK(result == correct);
    }

    SECTION("throws an error if subgraphs are not disjoint") {
      std::unordered_set<Node> src_subgraph = {n.at(0), n.at(1), n.at(2)};
      std::unordered_set<Node> dst_subgraph = {n.at(1), n.at(3)};
      CHECK_THROWS(
          get_edges_from_subgraph_to_subgraph(g, src_subgraph, dst_subgraph));
    }
  }
