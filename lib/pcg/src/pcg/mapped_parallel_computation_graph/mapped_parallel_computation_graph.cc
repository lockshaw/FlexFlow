#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/find_isomorphism_between_kwarg_dataflow_graphs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_kwarg_dataflow_graph_view_as_dot.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/rewrite_labelled_kwarg_dataflow_graph_node_labels.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_layer_attrs.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/materialize_labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<parallel_layer_guid_t> mpcg_get_parallel_layers(MappedParallelComputationGraph const &mpcg) {
  return get_parallel_layers(pcg_from_mpcg(mpcg));
}

std::optional<MappedOperatorTaskGroup> mpcg_get_mapping_for_layer(
  MappedParallelComputationGraph const &mpcg,
  parallel_layer_guid_t l)
{
  MappedParallelLayerAttrs layer_attrs = mpcg.raw_graph.at(l.raw_graph_node);

  return layer_attrs.mapping;
}

ParallelComputationGraph pcg_from_mpcg(MappedParallelComputationGraph const &mpcg) {
  LabelledKwargDataflowGraphView<ParallelLayerAttrs, ParallelTensorAttrs, TensorSlotName>
    raw_view = rewrite_labelled_kwarg_dataflow_graph_node_labels(
        mpcg.raw_graph,
        [](Node const &, MappedParallelLayerAttrs const &a) -> ParallelLayerAttrs {
          return unmapped_parallel_layer_attrs_from_mapped(a);
        });

  LabelledKwargDataflowGraph<ParallelLayerAttrs, ParallelTensorAttrs, TensorSlotName>
    raw_graph = materialize_labelled_kwarg_dataflow_graph_view(raw_view);

  return ParallelComputationGraph{
    raw_graph,
  };
}

MappedParallelComputationGraph mapped_pcg_from_pcg_and_mapped_op_task_groups(
  ParallelComputationGraph const &pcg,
  std::unordered_map<parallel_layer_guid_t, MappedOperatorTaskGroup> const &mapped_op_task_groups) 
{
  auto mapping_for_layer = [&](parallel_layer_guid_t l) 
    -> std::optional<MappedOperatorTaskGroup>
  {
    OperatorType op_type = get_op_type(pcg_get_op_attrs(pcg, l));

    if (should_be_mapped(op_type)) {
      return mapped_op_task_groups.at(l);
    } else {
      return std::nullopt;
    }
  };

  auto mpcg_layer_attrs_from_pcg_layer_attrs
    = [&](Node const &node, ParallelLayerAttrs const &pcg_layer_attrs)
        -> MappedParallelLayerAttrs
  {
    parallel_layer_guid_t l = parallel_layer_guid_t{node};

    return MappedParallelLayerAttrs{
      /*op_attrs=*/pcg_layer_attrs.op_attrs,
      /*name=*/pcg_layer_attrs.name,
      /*mapping=*/mapping_for_layer(l),
    };
  };

  LabelledKwargDataflowGraphView<MappedParallelLayerAttrs, ParallelTensorAttrs, TensorSlotName>
    result = 
      rewrite_labelled_kwarg_dataflow_graph_node_labels(pcg.raw_graph, mpcg_layer_attrs_from_pcg_layer_attrs);


  return MappedParallelComputationGraph{
    result,
  };
}

MappedParallelComputationGraph mapped_pcg_without_layer_names(MappedParallelComputationGraph const &mpcg) {
  LabelledKwargDataflowGraphView<MappedParallelLayerAttrs, ParallelTensorAttrs, TensorSlotName>
    result = 
      rewrite_labelled_kwarg_dataflow_graph_node_labels(
        mpcg.raw_graph,
        [&](Node const &, MappedParallelLayerAttrs const &with_name) -> MappedParallelLayerAttrs {
          return mapped_parallel_layer_attrs_without_layer_name(with_name);
        });

  return MappedParallelComputationGraph{
    result,
  };
}

std::string format_as(MappedParallelComputationGraph const &mapped_pcg) {
  return mapped_pcg_as_dot(mapped_pcg);
}

std::ostream &operator<<(std::ostream &s,
                         MappedParallelComputationGraph const &mapped_pcg) {
  return (s << fmt::to_string(mapped_pcg));
}

bool mapped_pcgs_are_isomorphic(MappedParallelComputationGraph const &src,
                                MappedParallelComputationGraph const &dst) {
  std::optional<bidict<Node, Node>> maybe_isomorphism = find_isomorphism_between_kwarg_dataflow_graphs(
     mapped_pcg_without_layer_names(src).raw_graph,
     mapped_pcg_without_layer_names(dst).raw_graph);

  return maybe_isomorphism.has_value();
}

std::string mapped_pcg_as_dot(MappedParallelComputationGraph const &mpcg) {

  std::function<std::string(MappedParallelLayerAttrs const &)> render_node_label =
      [](MappedParallelLayerAttrs const &a) -> std::string {
    RecordFormatter rr = mk_empty_record(Orientation::VERTICAL);

    RecordFormatter r = pcg_op_attrs_as_dot(a.op_attrs);

    if (a.name.has_value()) {
      rr << "Name" << a.name.value();
    }

    rr << r;

    rr << "Mapping";
    if (a.mapping.has_value()) {
      rr << mapped_operator_task_group_as_dot(a.mapping.value());
    } else {
      rr << "none";
    }

    std::ostringstream oss;
    RecordFormatter result = mk_empty_record(Orientation::HORIZONTAL);
    result << rr;
    oss << result;
    return oss.str();
  };

  std::function<std::string(ParallelTensorAttrs const &)> render_input_label =
      [](ParallelTensorAttrs const &a) -> std::string {
    RecordFormatter r = mk_empty_record(Orientation::HORIZONTAL);

    r << fmt::to_string(a.shape);

    std::ostringstream oss;
    oss << r;
    return oss.str();
  };

  std::function<std::string(TensorSlotName const &)> render_slot_name = [](TensorSlotName const &slot_name)
    -> std::string
  {
    return fmt::to_string(slot_name);
  };

  std::function<std::vector<TensorSlotName>(std::unordered_set<TensorSlotName> const &)> order_slots
    = [](std::unordered_set<TensorSlotName> const &slot_names)
      -> std::vector<TensorSlotName>
  {
    return sorted(slot_names);
  };

  return labelled_kwarg_dataflow_graph_view_as_dot(
      mpcg.raw_graph,
      render_node_label,
      render_input_label,
      render_slot_name,
      order_slots);
}

} // namespace FlexFlow
