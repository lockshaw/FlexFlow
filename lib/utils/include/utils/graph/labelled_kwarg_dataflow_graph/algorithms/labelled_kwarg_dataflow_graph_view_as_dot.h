#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_KWARG_DATAFLOW_GRAPH_VIEW_AS_DOT_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_as_dot.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename SlotName>
std::string labelled_kwarg_dataflow_graph_view_as_dot(
    LabelledKwargDataflowGraphView<NodeLabel,
                                   ValueLabel,
                                   SlotName> const &g,
    std::function<std::string(NodeLabel const &)> const &render_node_label,
    std::function<std::string(ValueLabel const &)> const &render_value_label,
    std::function<std::string(SlotName const &)> const &render_slot_name,
    std::function<std::vector<SlotName>(std::unordered_set<SlotName> const &)> const &order_slots)
{
  std::function<std::string(Node const &)> get_node_label = [&](Node const &n) -> std::string {
    return render_node_label(g.at(n));
  };

  return kwarg_dataflow_graph_as_dot(
    static_cast<KwargDataflowGraphView<SlotName>>(g),
    get_node_label,
    render_slot_name,
    order_slots);
}


} // namespace FlexFlow

#endif
