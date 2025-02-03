#include "utils/graph/open_dataflow_graph/algorithms/get_open_dataflow_graph_inputs.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_open_dataflow_graph_inputs(OpenDataflowGraphView)") {
    OpenDataflowGraph g =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    DataflowGraphInput i0 = g.add_input();
    DataflowGraphInput i1 = g.add_input();

    NodeAddedResult n0_added = g.add_node({}, 1_n);

    std::unordered_set<DataflowGraphInput> result =
        get_open_dataflow_graph_inputs(g);
    std::unordered_set<DataflowGraphInput> correct = {i0, i1};

    CHECK(result == correct);
  }
