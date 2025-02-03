#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_value_uses.h"
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_open_dataflow_value_uses(OpenDataflowGraphView, "
            "OpenDataflowValue)") {
    SECTION("value is a DataflowGraphInput") {
      OpenDataflowGraph g =
          OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

      DataflowGraphInput i0 = g.add_input();
      DataflowGraphInput i1 = g.add_input();

      NodeAddedResult n0_added = g.add_node(
          {OpenDataflowValue{i0}, OpenDataflowValue{i1}, OpenDataflowValue{i0}},
          1_n);
      Node n0 = n0_added.node;
      DataflowOutput o0 = get_only(n0_added.outputs);

      NodeAddedResult n1_added = g.add_node(
          {OpenDataflowValue{i1}, OpenDataflowValue{o0}, OpenDataflowValue{i0}},
          1_n);
      Node n1 = n1_added.node;

      std::unordered_set<DataflowInput> correct = {
          DataflowInput{n0, 0_n},
          DataflowInput{n0, 2_n},
          DataflowInput{n1, 2_n},
      };

      std::unordered_set<DataflowInput> result =
          get_open_dataflow_value_uses(g, OpenDataflowValue{i0});

      CHECK(result == correct);
    }

    SECTION("value is a DataflowOutput") {
      OpenDataflowGraph g =
          OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

      DataflowGraphInput i0 = g.add_input();

      NodeAddedResult n0_added = g.add_node({OpenDataflowValue{i0}}, 2_n);
      Node n0 = n0_added.node;
      DataflowOutput o0_0 = n0_added.outputs.at(0);
      DataflowOutput o0_1 = n0_added.outputs.at(1);

      NodeAddedResult n1_added = g.add_node({OpenDataflowValue{i0},
                                             OpenDataflowValue{o0_1},
                                             OpenDataflowValue{o0_0}},
                                            1_n);
      Node n1 = n1_added.node;

      NodeAddedResult n2_added =
          g.add_node({OpenDataflowValue{o0_1}, OpenDataflowValue{i0}}, 1_n);
      Node n2 = n2_added.node;

      std::unordered_set<DataflowInput> correct = {
          DataflowInput{n1, 1_n},
          DataflowInput{n2, 0_n},
      };

      std::unordered_set<DataflowInput> result =
          get_open_dataflow_value_uses(g, OpenDataflowValue{o0_1});

      CHECK(result == correct);
    }
  }
