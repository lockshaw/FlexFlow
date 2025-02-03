#include "utils/graph/open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/containers/get_only.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_graph_data.h"
#include "utils/graph/open_dataflow_graph/dataflow_input_edge_query.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_graph.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("permute_node_ids(OpenDataflowGraphView, bidict<NewNode, Node>)") {
    OpenDataflowGraph g =
        OpenDataflowGraph::create<UnorderedSetDataflowGraph>();

    DataflowGraphInput i0 = g.add_input();

    NodeAddedResult n0_added = g.add_node({OpenDataflowValue{i0}}, 1_n);
    Node n0 = n0_added.node;
    DataflowOutput n0_output = get_only(n0_added.outputs);

    NodeAddedResult n1_added =
        g.add_node({OpenDataflowValue{i0}, OpenDataflowValue{n0_output}}, 1_n);
    Node n1 = n1_added.node;
    DataflowOutput n1_output = get_only(n1_added.outputs);

    Node new_node0 = Node{5};
    Node new_node1 = Node{6};

    bidict<NewNode, Node> node_mapping = {
        {NewNode{new_node0}, n0},
        {NewNode{new_node1}, n1},
    };

    OpenDataflowGraphView result = permute_node_ids(g, node_mapping);
    OpenDataflowGraphData result_data = get_graph_data(result);

    OpenDataflowGraphData correct_data = OpenDataflowGraphData{
        {new_node0, new_node1},
        {
            OpenDataflowEdge{
                DataflowInputEdge{
                    i0,
                    DataflowInput{
                        new_node0,
                        0_n,
                    },
                },
            },
            OpenDataflowEdge{
                DataflowInputEdge{
                    i0,
                    DataflowInput{
                        new_node1,
                        0_n,
                    },
                },
            },
            OpenDataflowEdge{
                DataflowEdge{
                    DataflowOutput{
                        new_node0,
                        0_n,
                    },
                    DataflowInput{
                        new_node1,
                        1_n,
                    },
                },
            },
        },
        {i0},
        {
            DataflowOutput{
                new_node0,
                0_n,
            },
            DataflowOutput{
                new_node1,
                0_n,
            },
        },
    };

    CHECK(result_data == correct_data);

    // because get_graph_data only uses matchall nodes which don't require as
    // much updating, we also add test cases for the query methods with concrete
    // queries to check the through-node-permutation querying logic
    SECTION("query_nodes(NodeQuery)") {
      SECTION("check access to old nodes") {
        std::unordered_set<Node> result_nodes =
            result.query_nodes(NodeQuery{n0});
        std::unordered_set<Node> correct = {};
        CHECK(result_nodes == correct);
      }

      SECTION("check access to new nodes") {
        std::unordered_set<Node> result_nodes =
            result.query_nodes(NodeQuery{new_node0});
        std::unordered_set<Node> correct = {new_node0};
        CHECK(result_nodes == correct);
      }
    }

    SECTION("query_edges(OpenDataflowEdgeQuery)") {
      SECTION("check access to old edges") {
        OpenDataflowEdgeQuery query = OpenDataflowEdgeQuery{
            dataflow_input_edge_query_for_edge(
                DataflowInputEdge{i0, DataflowInput{n0, 0_n}}),
            dataflow_edge_query_for_edge(
                DataflowEdge{n0_output, DataflowInput{n1, 1_n}}),
        };
        std::unordered_set<OpenDataflowEdge> result_nodes =
            result.query_edges(query);
        std::unordered_set<OpenDataflowEdge> correct = {};
        CHECK(result_nodes == correct);
      }

      SECTION("check access to new edges") {
        DataflowEdge new_standard_edge = DataflowEdge{
            DataflowOutput{new_node0, 0_n},
            DataflowInput{new_node1, 1_n},
        };
        DataflowInputEdge new_input_edge = DataflowInputEdge{
            i0,
            DataflowInput{new_node0, 0_n},
        };
        OpenDataflowEdgeQuery query = OpenDataflowEdgeQuery{
            dataflow_input_edge_query_for_edge(new_input_edge),
            dataflow_edge_query_for_edge(new_standard_edge),
        };

        std::unordered_set<OpenDataflowEdge> result_nodes =
            result.query_edges(query);
        std::unordered_set<OpenDataflowEdge> correct = {
            OpenDataflowEdge{new_standard_edge},
            OpenDataflowEdge{new_input_edge},
        };

        CHECK(result_nodes == correct);
      }
    }

    SECTION("query_outputs(DataflowOutputQuery)") {
      SECTION("check access to old outputs") {
        DataflowOutput old_output = n0_output;

        DataflowOutputQuery query =
            dataflow_output_query_for_output(old_output);
        std::unordered_set<DataflowOutput> result_outputs =
            result.query_outputs(query);

        std::unordered_set<DataflowOutput> correct = {};

        CHECK(result_outputs == correct);
      }

      SECTION("check access to new outputs") {
        DataflowOutput new_output = DataflowOutput{new_node0, 0_n};

        DataflowOutputQuery query =
            dataflow_output_query_for_output(new_output);
        std::unordered_set<DataflowOutput> result_outputs =
            result.query_outputs(query);

        std::unordered_set<DataflowOutput> correct = {new_output};

        CHECK(result_outputs == correct);
      }
    }
  }
