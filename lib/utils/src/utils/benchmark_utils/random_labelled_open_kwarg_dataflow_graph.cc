#include "utils/benchmark_utils/random_labelled_open_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;

using RandomNodeLabel = std::function<NodeLabel(std::mt19937 &)>;
using RandomValueLabel = std::function<ValueLabel(std::mt19937 &)>;
using RandomGraphInputName = std::function<GraphInputName(std::mt19937 &)>;
using RandomSlotName = std::function<SlotName(std::mt19937 &)>;

template void random_labelled_open_kwarg_dataflow_graph(
    std::mt19937 &gen,
    LabelledOpenKwargDataflowGraph<NodeLabel,
                                   ValueLabel,
                                   GraphInputName,
                                   SlotName> &g,
    nonnegative_int num_nodes,
    nonnegative_int num_graph_inputs,
    nonnegative_int num_input_slots_per_node,
    nonnegative_int num_output_slots_per_node,
    RandomNodeLabel &&random_node_label,
    RandomValueLabel &&random_value_label,
    RandomGraphInputName &&random_graph_input_name,
    RandomSlotName &&random_slot_name);

} // namespace FlexFlow
