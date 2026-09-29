#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_TASK_SET_OPEN_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_INSTANCES_TASK_SET_OPEN_KWARG_DATAFLOW_GRAPH_H

#include "utils/containers/contains_key.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/extend.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_node_added_result.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/i_labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/overload.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values2.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
struct UnorderedSetLabelledOpenKwargDataflowGraph final
    : public ILabelledOpenKwargDataflowGraph<NodeLabel,
                                             ValueLabel,
                                             GraphInputName,
                                             SlotName>,
      public ILabelledKwargDataflowGraph<NodeLabel, ValueLabel, SlotName> {
public:
  UnorderedSetLabelledOpenKwargDataflowGraph() = default;

  KwargNodeAddedResult<SlotName>
      add_node(NodeLabel const &node_label,
               std::map<SlotName, KwargDataflowOutput<SlotName>> const &inputs,
               std::map<SlotName, ValueLabel> const &output_labels) override {
    return this->add_node(
        node_label,
        map_values(inputs,
                   [](KwargDataflowOutput<SlotName> const &o) {
                     return OpenKwargDataflowValue<GraphInputName, SlotName>{o};
                   }),
        output_labels);
  };

  KwargNodeAddedResult<SlotName> add_node(
      NodeLabel const &node_label,
      std::map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>> const
          &inputs,
      std::map<SlotName, ValueLabel> const &output_labels) override {
    Node new_node = this->node_source.new_node();
    this->nodes.insert({new_node, node_label});

    for (auto const &[input_slot_name, input_val] : inputs) {
      KwargDataflowInput<SlotName> dst = KwargDataflowInput<SlotName>{
          new_node,
          input_slot_name,
      };

      OpenKwargDataflowEdge<GraphInputName, SlotName> in_edge =
          mk_open_kwarg_dataflow_edge_from_src_val_and_dst(input_val, dst);

      this->edges.insert(in_edge);
    }

    auto mk_output = 
        [&](SlotName const &output_slot) -> KwargDataflowOutput<SlotName> {
          return KwargDataflowOutput<SlotName>{
              /*node=*/new_node,
              /*slot_name=*/output_slot,
          };
        };

    std::map<KwargDataflowOutput<SlotName>, ValueLabel> mapped_value_labels = 
      map_keys(output_labels, mk_output);

    this->outputs.insert({
      new_node, 
      mapped_value_labels,
    });

    extend(this->all_outputs, keys(mapped_value_labels));

    std::map<SlotName, KwargDataflowOutput<SlotName>> outputs = map_values2(
        output_labels,
        [&](SlotName const &output_slot, ValueLabel const &) -> KwargDataflowOutput<SlotName> {
          return mk_output(output_slot);
        });

    return KwargNodeAddedResult<SlotName>{
        /*node=*/new_node,
        /*outputs=*/outputs,
    };
  }

  KwargDataflowGraphInput<GraphInputName>
      add_input(GraphInputName const &name,
                ValueLabel const &value_label) override {
    KwargDataflowGraphInput<GraphInputName> input =
        KwargDataflowGraphInput{name};

    ASSERT(!contains_key(this->graph_inputs, input));
    this->graph_inputs.insert({input, value_label});

    return input;
  }

  std::set<Node> query_nodes(NodeQuery const &q) const override {
    return filter(keys(this->nodes),
                  [&](Node const &n) { return includes(q.nodes, n); });
  }

  std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
      query_edges(OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> const &q)
          const override {
    return filter(
        this->edges,
        [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
          return open_kwarg_dataflow_edge_query_includes(q, e);
        });
  }

  std::set<KwargDataflowOutput<SlotName>> query_outputs(
      KwargDataflowOutputQuery<SlotName> const &q) const override {

    bool is_all_nodes = is_matchall(q.nodes);
    bool is_all_slots = is_matchall(q.output_idxs);
    if (is_all_nodes && is_all_slots) {
      return this->all_outputs;
    } else if (is_all_slots) {
      std::set<KwargDataflowOutput<SlotName>> results;

      for (Node const &n : allowed_values(q.nodes)) {
        for (
          std::pair<KwargDataflowOutput<SlotName>, ValueLabel> const &m
          : this->outputs.at(n)
        ) { 
          results.emplace_hint(results.cend(), m.first);
        }
      }

      return results;
    } else {
      return filter(this->all_outputs,
                    [&](KwargDataflowOutput<SlotName> const &output) {
                      return kwarg_dataflow_output_query_includes(q, output);
                    });
    }
  }

  std::set<KwargDataflowGraphInput<GraphInputName>>
      get_inputs() const override {
    return keys(this->graph_inputs);
  }

  NodeLabel at(Node const &n) const override {
    return this->nodes.at(n);
  }

  ValueLabel at(OpenKwargDataflowValue<GraphInputName, SlotName> const &v)
      const override {
    return v.template visit<ValueLabel>(overload{
        [&](KwargDataflowOutput<SlotName> const &o) -> ValueLabel {
          return this->outputs.at(o.node).at(o);
        },
        [&](KwargDataflowGraphInput<GraphInputName> const &gi) -> ValueLabel {
          return this->graph_inputs.at(gi);
        }});
  }

  void inplace_materialize_from(
      LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const
          &view) override {
    std::set<Node> view_nodes = get_nodes(view);
    std::set<KwargDataflowEdge<SlotName>> view_edges =
        get_all_kwarg_dataflow_edges(view);
    std::set<KwargDataflowOutput<SlotName>> view_outputs =
        get_all_kwarg_dataflow_outputs(view);

    this->graph_inputs.clear();
    this->nodes =
        generate_map(view_nodes, [&](Node const &n) { return view.at(n); });

    this->edges =
        transform(view_edges,
                  [&](KwargDataflowEdge<SlotName> const &e)
                      -> OpenKwargDataflowEdge<GraphInputName, SlotName> {
                    return OpenKwargDataflowEdge<GraphInputName, SlotName>{e};
                  });

    this->outputs.clear();
    for (Node const &n : view_nodes) { 
      this->outputs.insert({n, {}});
    }
    for (KwargDataflowOutput<SlotName> const &o : view_outputs) { 
      this->outputs.at(o.node).insert({o, view.at(o)});
    }

    this->all_outputs = view_outputs;
  }

  void inplace_materialize_from(
      LabelledOpenKwargDataflowGraphView<NodeLabel,
                                         ValueLabel,
                                         GraphInputName,
                                         SlotName> const &view) override {
    std::set<KwargDataflowGraphInput<GraphInputName>> view_inputs =
        get_all_kwarg_dataflow_graph_inputs(view);
    std::set<Node> view_nodes = get_nodes(view);
    std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> view_edges =
        get_all_open_kwarg_dataflow_edges(view);
    std::set<KwargDataflowOutput<SlotName>> view_outputs =
        get_all_kwarg_dataflow_outputs(view);

    this->graph_inputs = generate_map(
        view_inputs, [&](KwargDataflowGraphInput<GraphInputName> const &i) {
          return view.at(OpenKwargDataflowValue<GraphInputName, SlotName>{i});
        });
    this->nodes =
        generate_map(view_nodes, [&](Node const &n) { return view.at(n); });

    this->edges = view_edges;

    this->outputs.clear();
    for (Node const &n : view_nodes) { 
      this->outputs.insert({n, {}});
    }
    for (KwargDataflowOutput<SlotName> const &o : view_outputs) { 
      this->outputs.at(o.node).insert({o, view.at(OpenKwargDataflowValue<GraphInputName, SlotName>{o})});
    }

    this->all_outputs = view_outputs;
  }

  UnorderedSetLabelledOpenKwargDataflowGraph *clone() const override {
    return new UnorderedSetLabelledOpenKwargDataflowGraph{
        this->node_source,
        this->graph_inputs,
        this->nodes,
        this->edges,
        this->outputs,
        this->all_outputs,
    };
  }

private:
  UnorderedSetLabelledOpenKwargDataflowGraph(
      NodeSource const &node_source,
      std::map<KwargDataflowGraphInput<GraphInputName>, ValueLabel> const
          &graph_inputs,
      std::map<Node, NodeLabel> const &nodes,
      std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> const &edges,
      std::map<Node, std::map<KwargDataflowOutput<SlotName>, ValueLabel>> const &outputs,
      std::set<KwargDataflowOutput<SlotName>> const &all_outputs)
      : node_source(node_source), graph_inputs(graph_inputs), nodes(nodes),
        edges(edges), outputs(outputs) {}

private:
  NodeSource node_source;

  std::map<KwargDataflowGraphInput<GraphInputName>, ValueLabel> graph_inputs;
  std::map<Node, NodeLabel> nodes;
  std::set<OpenKwargDataflowEdge<GraphInputName, SlotName>> edges;
  std::map<Node, std::map<KwargDataflowOutput<SlotName>, ValueLabel>> outputs;
  std::set<KwargDataflowOutput<SlotName>> all_outputs;
};

} // namespace FlexFlow

#endif
