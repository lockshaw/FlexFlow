#include "utils/graph/digraph/algorithms/get_weakly_connected_components.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_weakly_connected_components") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SECTION("single node") {
      std::vector<Node> n = add_nodes(g, 1);

      std::unordered_set<std::unordered_set<Node>> result =
          get_weakly_connected_components(g);
      std::unordered_set<std::unordered_set<Node>> correct = {{n.at(0)}};
      CHECK(result == correct);
    }

    SECTION("one node cycle") {
      std::vector<Node> n = add_nodes(g, 1);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(0)},
                });

      std::unordered_set<std::unordered_set<Node>> result =
          get_weakly_connected_components(g);
      std::unordered_set<std::unordered_set<Node>> correct = {{n.at(0)}};
      CHECK(result == correct);
    }

    SECTION("two disconnected nodes with cycles") {
      std::vector<Node> n = add_nodes(g, 2);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(0)},
                    DirectedEdge{n.at(1), n.at(1)},
                });

      std::unordered_set<std::unordered_set<Node>> result =
          get_weakly_connected_components(g);
      std::unordered_set<std::unordered_set<Node>> correct = {{n.at(0)},
                                                              {n.at(1)}};
      CHECK(result == correct);
    }

    SECTION("two unidirectionally connected nodes") {
      std::vector<Node> n = add_nodes(g, 2);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                });

      std::unordered_set<std::unordered_set<Node>> result =
          get_weakly_connected_components(g);
      std::unordered_set<std::unordered_set<Node>> correct = {
          {n.at(0), n.at(1)}};
      CHECK(result == correct);
    }

    SECTION("two node cycle") {
      std::vector<Node> n = add_nodes(g, 2);
      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(1), n.at(0)},
                });

      std::unordered_set<std::unordered_set<Node>> result =
          get_weakly_connected_components(g);
      std::unordered_set<std::unordered_set<Node>> correct = {
          {n.at(0), n.at(1)}};
      CHECK(result == correct);
    }

    SECTION("nontrivial graph") {
      std::vector<Node> n = add_nodes(g, 5);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(1)},
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(1), n.at(0)},
                    DirectedEdge{n.at(0), n.at(0)},
                    DirectedEdge{n.at(3), n.at(4)},
                    DirectedEdge{n.at(4), n.at(4)},
                    DirectedEdge{n.at(4), n.at(3)},
                });

      std::unordered_set<std::unordered_set<Node>> result =
          get_weakly_connected_components(g);
      std::unordered_set<std::unordered_set<Node>> correct = {
          {n.at(0), n.at(1), n.at(2)},
          {n.at(3), n.at(4)},
      };
      CHECK(result == correct);
    }
  }
