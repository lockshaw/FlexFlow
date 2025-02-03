#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_outgoing_edges(DataflowGraphView, Node)") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1_n);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({o1}, 1_n);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({o1}, 1_n);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o2}, 1_n);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    SECTION("n2 - single outgoing edge") {
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, n2);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{o2, DataflowInput{n4, 0_n}},
      };
      CHECK(result == correct);
    }

    SECTION("n1 - multiple outgoing edges") {
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, n1);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{o1, DataflowInput{n2, 0_n}},
          DataflowEdge{o1, DataflowInput{n3, 0_n}},
      };
      CHECK(result == correct);
    }

    SECTION("n4 - no outgoing edges") {
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, n4);
      std::unordered_set<DataflowEdge> correct = {};
      CHECK(result == correct);
    }
  }

  TEST_CASE("get_outgoing_edges(DataflowGraphView, std::unordered_set<Node>)") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1_n);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({o1}, 1_n);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({o1}, 1_n);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o2}, 1_n);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    SECTION("multiple nodes - combined outgoing edges") {
      std::unordered_set<Node> nodes = {n1, n2};
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, nodes);
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{o1, DataflowInput{n2, 0_n}},
          DataflowEdge{o1, DataflowInput{n3, 0_n}},
          DataflowEdge{o2, DataflowInput{n4, 0_n}},
      };
      CHECK(result == correct);
    }

    SECTION("multiple nodes - no outgoing edges") {
      std::unordered_set<Node> nodes = {n3, n4};
      std::unordered_set<DataflowEdge> result = get_outgoing_edges(g, nodes);
      std::unordered_set<DataflowEdge> correct = {};
      CHECK(result == correct);
    }
  }
