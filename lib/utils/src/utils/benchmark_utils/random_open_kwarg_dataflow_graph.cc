#include "utils/benchmark_utils/random_open_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;
using RandomGraphInputName = std::function<GraphInputName(std::mt19937 &)>;
using RandomSlotName = std::function<SlotName(std::mt19937 &)>;

template void random_open_kwarg_dataflow_graph(
    std::mt19937 &gen,
    OpenKwargDataflowGraph<GraphInputName, SlotName> &g,
    nonnegative_int num_nodes,
    nonnegative_int num_graph_inputs,
    nonnegative_int num_input_slots_per_node,
    nonnegative_int num_output_slots_per_node,
    RandomGraphInputName &&random_graph_input_name,
    RandomSlotName &&random_slot_name);

} // namespace FlexFlow
