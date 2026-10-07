#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/benchmark_utils/random_set.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/vector_split.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/random_utils.h"
#include <random>

namespace FlexFlow {

template <
    typename RandomGraphInputName,
    typename RandomSlotName,
    typename GraphInputName =
        std::invoke_result_t<RandomGraphInputName, std::mt19937 &>,
    typename SlotName = std::invoke_result_t<RandomSlotName, std::mt19937 &>>
void random_open_kwarg_dataflow_graph(
    std::mt19937 &gen,
    OpenKwargDataflowGraph<GraphInputName, SlotName> &g,
    nonnegative_int num_nodes,
    nonnegative_int num_graph_inputs,
    nonnegative_int num_input_slots_per_node,
    nonnegative_int num_output_slots_per_node,
    RandomGraphInputName &&random_graph_input_name,
    RandomSlotName &&random_slot_name) {
  std::set<GraphInputName> graph_input_names =
      random_set(gen, num_graph_inputs, random_graph_input_name);

  std::set<KwargDataflowGraphInput<GraphInputName>> graph_inputs =
      transform(graph_input_names,
                [&](GraphInputName const &graph_input_name)
                    -> KwargDataflowGraphInput<GraphInputName> {
                  return g.add_input(graph_input_name);
                });

  std::set<OpenKwargDataflowValue<GraphInputName, SlotName>> graph_values =
      transform(graph_inputs,
                [](KwargDataflowGraphInput<GraphInputName> const &graph_input)
                    -> OpenKwargDataflowValue<GraphInputName, SlotName> {
                  return OpenKwargDataflowValue<GraphInputName, SlotName>{
                      graph_input};
                });

  for (nonnegative_int _ : nonnegative_range(num_nodes)) {
    std::vector<SlotName> slot_names = vector_of(
        random_set(gen,
                   num_input_slots_per_node + num_output_slots_per_node,
                   random_slot_name));

    auto [input_slot_name_vec, output_slot_name_vec] = vector_split(
        slot_names, num_input_slots_per_node.int_from_nonnegative_int());

    std::set<SlotName> input_slot_names = set_of(input_slot_name_vec);
    std::set<SlotName> output_slot_names = set_of(output_slot_name_vec);

    std::map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>>
        node_inputs = generate_map(
            input_slot_names,
            [&](SlotName const &input_slot_names)
                -> OpenKwargDataflowValue<GraphInputName, SlotName> {
              return select_random(gen, graph_values);
            });

    KwargNodeAddedResult<SlotName> added =
        g.add_node(node_inputs, output_slot_names);
    for (KwargDataflowOutput<SlotName> const &output : values(added.outputs)) {
      graph_values.insert(
          OpenKwargDataflowValue<GraphInputName, SlotName>{output});
    }
  }
}

} // namespace FlexFlow

#endif
