#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/containers/set_of.h"
#include "utils/graph/instances/unordered_set_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphView<GraphInputName, SlotName>
    view_from_open_kwarg_dataflow_graph_data(
        OpenKwargDataflowGraphData<GraphInputName, SlotName> const &data) {
  require_open_kwarg_dataflow_graph_data_is_valid(data);

  return OpenKwargDataflowGraphView<GraphInputName, SlotName>::template create<
      UnorderedSetOpenKwargDataflowGraph<GraphInputName, SlotName>>(data);
}

} // namespace FlexFlow

#endif
