#include "utils/graph/dataflow_graph/algorithms/get_incoming_edges.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_incoming_edges(DataflowGraphView, Node)") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1_n);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({}, 1_n);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({o2}, 1_n);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o2, o3}, 1_n);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    SECTION("n4 - multiple incoming edges") {
      std::vector<DataflowEdge> result = get_incoming_edges(g, n4);
      std::vector<DataflowEdge> correct = {
          DataflowEdge{o2, DataflowInput{n4, 0_n}},
          DataflowEdge{o3, DataflowInput{n4, 1_n}}};
      CHECK(result == correct);
    }

    SECTION("n3- single incoming edge") {
      std::vector<DataflowEdge> result = get_incoming_edges(g, n3);
      std::vector<DataflowEdge> correct = {
          DataflowEdge{o2, DataflowInput{n3, 0_n}},
      };
      CHECK(result == correct);
    }

    SECTION("n1- no incoming edges") {
      std::vector<DataflowEdge> result = get_incoming_edges(g, n1);
      std::vector<DataflowEdge> correct = {};
      CHECK(result == correct);
    }
  }
