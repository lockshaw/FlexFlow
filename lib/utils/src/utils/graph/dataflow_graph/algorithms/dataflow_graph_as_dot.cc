#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_as_dot.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_keys.h"
#include "utils/dot_file.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms/view_as_open_dataflow_graph.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/with_labelling.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/render_dot.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

std::string dataflow_graph_as_dot(
  DataflowGraphView const &g,
  std::optional<std::function<std::string(Node const &)>> const &get_node_label,
  std::optional<std::function<std::string(DataflowInput const &)>> const &get_input_label,
  std::optional<std::function<std::string(DataflowOutput const &)>> const &get_output_label)
{
  std::ostringstream oss;
  DotFile<std::string> dot{oss};

  dataflow_graph_as_dot(dot, g, get_node_label, get_input_label, get_output_label);

  dot.close();

  return oss.str();
}

void dataflow_graph_as_dot(
  DotFile<std::string> &dot,
  DataflowGraphView const &g,
  std::optional<std::function<std::string(Node const &)>> const &get_node_label,
  std::optional<std::function<std::string(DataflowInput const &)>> const &get_input_label,
  std::optional<std::function<std::string(DataflowOutput const &)>> const &get_output_label)
{

  auto get_node_name = [](Node n) { return fmt::format("n{}", n.raw_uid); };

  std::function<std::string(Node const &)> resolved_get_node_label =
    get_node_label.value_or(get_node_name);

  std::function<std::string(DataflowInput const &)> resolved_get_input_label =
    get_input_label.value_or([](DataflowInput const &i) { return fmt::to_string(i.idx); });

  std::function<std::string(DataflowOutput const &)> resolved_get_output_label =
    get_output_label.value_or([](DataflowOutput const &o) { return fmt::to_string(o.idx); });

  auto get_input_field = [](nonnegative_int idx) {
    return fmt::format("i{}", idx);
  };

  auto get_output_field = [](nonnegative_int idx) {
    return fmt::format("o{}", idx);
  };

  for (Node const &n : get_nodes(g)) {
    std::vector<DataflowInput> n_inputs = get_dataflow_inputs(g, n);
    std::vector<DataflowOutput> n_outputs = get_outputs(g, n);

    RecordFormatter inputs_record = mk_empty_record(Orientation::HORIZONTAL);
    for (DataflowInput const &i : n_inputs) {
      inputs_record << fmt::format("<{}>{}", get_input_field(i.idx), resolved_get_input_label(i));
    }

    RecordFormatter outputs_record = mk_empty_record(Orientation::HORIZONTAL);
    for (DataflowOutput const &o : n_outputs) {
      outputs_record << fmt::format("<{}>{}", get_output_field(o.idx), resolved_get_output_label(o));
    }

    RecordFormatter rec = mk_empty_record(Orientation::VERTICAL);
    rec << inputs_record << resolved_get_node_label(n) << outputs_record;

    dot.add_record_node(get_node_name(n), rec);
  }

  for (DataflowEdge const &e : get_edges(g)) {
    dot.add_edge(get_node_name(e.src.node),
                 get_node_name(e.dst.node),
                 get_output_field(e.src.idx),
                 get_input_field(e.dst.idx));
  }
}

} // namespace FlexFlow
