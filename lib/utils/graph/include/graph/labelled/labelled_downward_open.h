#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_OPEN_H

#include "labelled_downward_open_interfaces.h"
#include "labelled_open.fwd.h"
#include "utils/graph/internal.h"
#include "labelled_downward_open.decl.h"

namespace FlexFlow {

// struct LabelledDownwardOpenMultiDiGraphView;

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
template <typename InputLabel>
LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>::operator LabelledOpenMultiDiGraphView<NodeLabel,
                                      EdgeLabel,
                                      InputLabel,
                                      OutputLabel>() const {
  return GraphInternal::create_labelled_open_multidigraph_view(this->ptr);
}

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  OutputLabel const &LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>::at(OutputMultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  EdgeLabel const &LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>::at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  NodeLabel const &LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>::at(Node const &n) const {
    return this->ptr->at(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  std::unordered_set<Node> LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>::query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  std::unordered_set<DownwardOpenMultiDiEdge>
      LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>::query_edges(DownwardOpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  template <typename BaseImpl, typename... Args>
  typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledDownwardOpenMultiDiGraphView>::type
      LabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>::create(Args &&...args) {
    return LabelledDownwardOpenMultiDiGraphView(
        std::make_shared<BaseImpl const>(std::forward<Args>(args)...));
  }


// struct LabelledDownardOpenMultiDiGraph;

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  OutputLabel const &LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::at(OutputMultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  OutputLabel &LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::at(OutputMultiDiEdge const &e) {
    return this->ptr.get_mutable()->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  EdgeLabel const &LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  EdgeLabel &LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::at(MultiDiEdge const &e) {
    return this->ptr.get_mutable()->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  NodeLabel const &LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::at(Node const &n) const {
    return this->ptr->at(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  NodeLabel &LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::at(Node const &n) {
    return this->ptr.get_mutable()->at(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  std::unordered_set<Node> LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  std::unordered_set<DownwardOpenMultiDiEdge>
      LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::query_edges(DownwardOpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  Node LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::add_node(NodeLabel const &l) {
    return this->ptr.get_mutable()->add_node(l);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  void LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::add_node_unsafe(Node const &n) {
    return this->ptr.get_mutable()->add_node_unsafe(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  void LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::remove_node_unsafe(Node const &n) {
    return this->ptr.get_mutable()->remove_node_unsafe(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  void LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::add_edge(MultiDiEdge const &e, EdgeLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  void LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::add_edge(OutputMultiDiEdge const &e, OutputLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  void LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::remove_edge(OutputMultiDiEdge const &e) {
    return this->ptr.get_mutable()->remove_edge(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  void LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::remove_edge(MultiDiEdge const &e) {
    return this->ptr.get_mutable()->remove_edge(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledDownardOpenMultiDiGraph>::type
      LabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>::create() {
    return LabelledDownardOpenMultiDiGraph(make_unique<BaseImpl>());
  }

} // namespace FlexFlow

#endif
