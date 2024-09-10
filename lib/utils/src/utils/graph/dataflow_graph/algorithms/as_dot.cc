#include "utils/graph/dataflow_graph/algorithms/as_dot.h"
#include "utils/containers/generate_map.h"
#include "utils/dot_file.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/view_as_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/with_labelling.h"
#include "utils/graph/node/algorithms.h"
#include "utils/record_formatter.h"
#include "utils/render_dot.h"
#include "utils/containers/map_keys.h"

namespace FlexFlow {

std::string as_dot(DataflowGraphView const &g) {
  auto get_node_attrs = [](Node const &) {
    return std::unordered_map<std::string, std::string>{}; 
  };

  std::unordered_map<Node, std::unordered_map<std::string, std::string>> node_labels = generate_map(get_nodes(g), get_node_attrs);

  auto get_output_label = [](DataflowOutput const &o) {
    return fmt::to_string(o.idx);
  };

  std::unordered_map<DataflowOutput, std::string> output_labels = generate_map(get_all_dataflow_outputs(g), get_output_label);
  std::unordered_map<OpenDataflowValue, std::string> value_labels = map_keys(output_labels, [](DataflowOutput const &o) { return OpenDataflowValue{o}; });

  return render_dot(with_labelling(view_as_open_dataflow_graph(g), node_labels, value_labels));
}

} // namespace FlexFlow
