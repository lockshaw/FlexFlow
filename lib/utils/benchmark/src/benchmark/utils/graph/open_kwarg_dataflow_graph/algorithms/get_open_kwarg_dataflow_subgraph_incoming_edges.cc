#include "benchmark/utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_incoming_edges.h"
#include "utils/benchmark_utils/loop.h"
#include "utils/benchmark_utils/random_hex_string.h"
#include "utils/benchmark_utils/random_int.h"
#include "utils/benchmark_utils/random_labelled_open_kwarg_dataflow_graph.h"
#include "utils/benchmark_utils/random_open_kwarg_dataflow_graph.h"
#include "utils/benchmark_utils/random_subset.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_incoming_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph.h"

namespace FlexFlow {

void benchmark_get_open_kwarg_dataflow_subgraph_incoming_edges_unlabelled(
    bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  OpenKwargDataflowGraph<int, std::string> g =
      OpenKwargDataflowGraph<int, std::string>::template create<
          UnorderedSetOpenKwargDataflowGraph<int, std::string>>();

  auto random_graph_input_name = [](std::mt19937 &gen) -> int {
    return random_int(gen);
  };

  auto random_slot_name = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 8_n);
  };

  random_open_kwarg_dataflow_graph(
      gen,
      g,
      /*num_nodes=*/100_n,
      /*num_graph_inputs=*/20_n,
      /*num_input_slots_per_node=*/3_n,
      /*num_output_slots_per_node=*/2_n,
      /*random_graph_input_name=*/random_graph_input_name,
      /*random_slot_name=*/random_slot_name);

  std::set<Node> subgraph = random_subset(gen, 4_n, get_nodes(g));

  LOOP(10, dry_run) {
    get_open_kwarg_dataflow_subgraph_incoming_edges(g, subgraph);
  }
}

void benchmark_get_open_kwarg_dataflow_subgraph_incoming_edges_labelled(
    bool dry_run) {
  std::mt19937 gen;
  gen.seed(0);

  LabelledOpenKwargDataflowGraph<std::string, std::string, int, std::string> g =
      LabelledOpenKwargDataflowGraph<std::string,
                                     std::string,
                                     int,
                                     std::string>::
          template create<
              UnorderedSetLabelledOpenKwargDataflowGraph<std::string,
                                                         std::string,
                                                         int,
                                                         std::string>>();

  auto random_node_label = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 7_n);
  };

  auto random_value_label = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 9_n);
  };

  auto random_graph_input_name = [](std::mt19937 &gen) -> int {
    return random_int(gen);
  };

  auto random_slot_name = [](std::mt19937 &gen) -> std::string {
    return random_hex_string(gen, 8_n);
  };

  random_labelled_open_kwarg_dataflow_graph(
      gen,
      g,
      /*num_nodes=*/100_n,
      /*num_graph_inputs=*/20_n,
      /*num_input_slots_per_node=*/3_n,
      /*num_output_slots_per_node=*/2_n,
      /*random_node_label=*/random_node_label,
      /*random_value_label=*/random_value_label,
      /*random_graph_input_name=*/random_graph_input_name,
      /*random_slot_name=*/random_slot_name);

  std::set<Node> subgraph = random_subset(gen, 4_n, get_nodes(g));

  LOOP(10, dry_run) {
    get_open_kwarg_dataflow_subgraph_incoming_edges(g, subgraph);
  }
}

} // namespace FlexFlow
