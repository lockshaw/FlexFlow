#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/complete_bipartite_composite/is_complete_bipartite_digraph.h"
#include "utils/graph/digraph/digraph.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("is_complete_bipartite_digraph(UndirectedGraphView, "
            "std::unordered_set)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SECTION("simple bipartite graph") {
      std::vector<Node> n = add_nodes(g, 5);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(0), n.at(4)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(1), n.at(4)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                });

      SECTION("source group") {
        std::unordered_set<Node> group1 = {n.at(0), n.at(1), n.at(2)};

        bool result = is_complete_bipartite_digraph(g, group1);
        bool correct = true;

        CHECK(result == correct);
      }

      SECTION("sink group") {
        std::unordered_set<Node> group1 = {n.at(3), n.at(4)};

        bool result = is_complete_bipartite_digraph(g, group1);
        bool correct = false;

        CHECK(result == correct);
      }
    }

    SECTION("missing an edge (i.e., not complete)") {
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(1), n.at(3)},
                });
      std::unordered_set<Node> group1 = {n.at(0), n.at(1)};

      bool result = is_complete_bipartite_digraph(g, group1);
      bool correct = false;

      CHECK(result == correct);
    }

    SECTION("extra edge (i.e., not bipartite)") {
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(3)},
                });
      std::unordered_set<Node> group1 = {n.at(0), n.at(1)};

      bool result = is_complete_bipartite_digraph(g, group1);
      bool correct = false;

      CHECK(result == correct);
    }

    SECTION("flipped edge") {
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(2), n.at(1)},
                    DirectedEdge{n.at(1), n.at(3)},
                });
      std::unordered_set<Node> group1 = {n.at(0), n.at(1)};

      bool result = is_complete_bipartite_digraph(g, group1);
      bool correct = false;

      CHECK(result == correct);
    }

    SECTION("group too small") {
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                });
      std::unordered_set<Node> group1 = {n.at(0)};

      bool result = is_complete_bipartite_digraph(g, group1);
      bool correct = false;

      CHECK(result == correct);
    }
  }

  TEST_CASE("is_complete_bipartite_digraph(UndirectedGraphView)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SECTION("simple bipartite graph") {
      std::vector<Node> n = add_nodes(g, 5);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(0), n.at(4)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(1), n.at(4)},
                    DirectedEdge{n.at(2), n.at(3)},
                    DirectedEdge{n.at(2), n.at(4)},
                });

      bool result = is_complete_bipartite_digraph(g);
      bool correct = true;

      CHECK(result == correct);
    }

    SECTION("missing an edge") {
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(1), n.at(3)},
                });

      bool result = is_complete_bipartite_digraph(g);
      bool correct = false;

      CHECK(result == correct);
    }

    SECTION("extra edge") {
      std::vector<Node> n = add_nodes(g, 4);

      add_edges(g,
                {
                    DirectedEdge{n.at(0), n.at(2)},
                    DirectedEdge{n.at(0), n.at(3)},
                    DirectedEdge{n.at(1), n.at(2)},
                    DirectedEdge{n.at(1), n.at(3)},
                    DirectedEdge{n.at(2), n.at(3)},
                });

      bool result = is_complete_bipartite_digraph(g);
      bool correct = false;

      CHECK(result == correct);
    }
  }
