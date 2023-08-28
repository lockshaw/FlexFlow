#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_H

#include "labelled_upward_open.decl.h"
#include "utils/graph/internal.h"

namespace FlexFlow {

// struct LabelledUpwardOpenMultiDiGraphView

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
template <typename OutputLabel>
LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::
operator LabelledOpenMultiDiGraphView<NodeLabel,
                                      EdgeLabel,
                                      InputLabel,
                                      OutputLabel>() const {
  return GraphInternal::create_labelled_upward_open_multidigraph_view(this->ptr);
}

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
InputLabel const &LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::at(InputMultiDiEdge const &e) const {
  return this->ptr->at(e);
}

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  EdgeLabel const &LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  NodeLabel const &LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::at(Node const &n) const {
    return this->ptr->at(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  std::unordered_set<Node> LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  std::unordered_set<UpwardOpenMultiDiEdge>
      LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::query_edges(UpwardOpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  template <typename BaseImpl, typename... Args>
  typename std::enable_if<std::is_base_of<
    ILabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>, BaseImpl>::value,
                                 LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>>::type
      LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::create(Args &&...args) {
    return LabelledUpwardOpenMultiDiGraphView(
        std::make_shared<BaseImpl const>(std::forward<Args>(args)...));
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  LabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>::LabelledUpwardOpenMultiDiGraphView(
      std::shared_ptr<ILabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel> const> ptr)
      : ptr(std::move(ptr)) {}


// struct LabelledUpwardOpenMultiDiGraph

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
template <typename OutputLabel>
LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::operator LabelledOpenMultiDiGraphView<NodeLabel,
                                      EdgeLabel,
                                      InputLabel,
                                      OutputLabel>() const {
  NOT_IMPLEMENTED();
}

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  InputLabel const &LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::at(InputMultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  InputLabel &LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::at(InputMultiDiEdge const &e) {
    return this->ptr.get_mutable()->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  EdgeLabel const &LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::at(MultiDiEdge const &e) const {
    return this->ptr->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  EdgeLabel &LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::at(MultiDiEdge const &e) {
    return this->ptr.get_mutable()->at(e);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  NodeLabel const &LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::at(Node const &n) const {
    return this->ptr->at(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  NodeLabel &LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::at(Node const &n) {
    return this->ptr.get_mutable()->at(n);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  std::unordered_set<Node> LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  std::unordered_set<UpwardOpenMultiDiEdge>
      LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::query_edges(UpwardOpenMultiDiEdgeQuery const &q) const {
    return this->ptr->query_edges(q);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  Node LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::add_node(NodeLabel const &l) {
    return this->ptr.get_mutable()->add_node(l);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  void LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::add_node_unsafe(Node const &n, NodeLabel const &l) {
    return this->ptr.get_mutable()->add_node_unsafe();
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  void LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::add_edge(MultiDiEdge const &e, EdgeLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  void LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::add_edge(InputMultiDiEdge const &e, InputLabel const &l) {
    return this->ptr.get_mutable()->add_edge(e, l);
  }

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
  template <typename BaseImpl>
  typename std::enable_if<std::is_base_of<
    ILabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>
  , BaseImpl>::value,
                                 LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>>::type
      LabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>::create() {
    return LabelledUpwardOpenMultiDiGraph{make_unique<BaseImpl>()};
  }

} // namespace FlexFlow

#endif
