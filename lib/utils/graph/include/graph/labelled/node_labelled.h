#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_H

#include "node_labelled.decl.h"
#include "node_labelled_interfaces.h"
#include "utils/graph/internal.h"

namespace FlexFlow {

// struct NodeLabelledMultiDiGraphView

template <typename NodeLabel>
NodeLabelledMultiDiGraphView<NodeLabel>::operator MultiDiGraphView() const {
  return GraphInternal::create_multidigraphview(this->ptr);
}

template <typename NodeLabel>
NodeLabel const &NodeLabelledMultiDiGraphView<NodeLabel>::at(Node const &n) const {
  return this->ptr->at(n);
}

template <typename NodeLabel>
std::unordered_set<Node> NodeLabelledMultiDiGraphView<NodeLabel>::query_nodes(NodeQuery const &q) const {
  return this->ptr->query_nodes(q);
}

template <typename NodeLabel>
std::unordered_set<MultiDiEdge> NodeLabelledMultiDiGraphView<NodeLabel>::query_edges(MultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

template <typename NodeLabel>
template <typename BaseImpl, typename... Args>
typename std::enable_if<std::is_base_of<INodeLabelledMultiDiGraphView<NodeLabel>, BaseImpl>::value,
                               NodeLabelledMultiDiGraphView<NodeLabel>>::type
    NodeLabelledMultiDiGraphView<NodeLabel>::create(Args &&...args) {
  return NodeLabelledMultiDiGraphView(
      std::make_shared<BaseImpl>(std::forward<Args>(args)...));
}

// struct NodeLabelledMultiDiGraph

template <typename NodeLabel>
NodeLabelledMultiDiGraph<NodeLabel>::operator NodeLabelledMultiDiGraphView<NodeLabel>() const {
  return GraphInternal::create_node_labelled_multidigraph_view(this->ptr);
}


template <typename NodeLabel>
void swap(NodeLabelledMultiDiGraph<NodeLabel> &lhs,
                 NodeLabelledMultiDiGraph<NodeLabel> &rhs) {
  using std::swap;

  swap(lhs.ptr, rhs.ptr);
}

template <typename NodeLabel>
Node NodeLabelledMultiDiGraph<NodeLabel>::add_node(NodeLabel const &l) {
  return this->ptr->add_node(l);
}

template <typename NodeLabel>
NodeLabel &NodeLabelledMultiDiGraph<NodeLabel>::at(Node const &n) {
  return this->ptr->at(n);
}

template <typename NodeLabel>
NodeLabel const &NodeLabelledMultiDiGraph<NodeLabel>::at(Node const &n) const {
  return this->ptr->at(n);
}

template <typename NodeLabel>
void NodeLabelledMultiDiGraph<NodeLabel>::add_edge(MultiDiEdge const &e) {
  return this->ptr->add_edge(e);
}

template <typename NodeLabel>
  std::unordered_set<Node> NodeLabelledMultiDiGraph<NodeLabel>::query_nodes(NodeQuery const &q) const {
    return this->ptr->query_nodes(q);
  }

template <typename NodeLabel>
std::unordered_set<MultiDiEdge> NodeLabelledMultiDiGraph<NodeLabel>::query_edges(MultiDiEdgeQuery const &q) const {
  return this->ptr->query_edges(q);
}

template <typename NodeLabel>
template <typename BaseImpl>
typename std::enable_if<std::is_base_of<INodeLabelledMultiDiGraph<NodeLabel>, BaseImpl>::value,
                               NodeLabelledMultiDiGraph<NodeLabel>>::type
    NodeLabelledMultiDiGraph<NodeLabel>::create() {
  return NodeLabelledMultiDiGraph(make_unique<BaseImpl>());
}

template <typename NodeLabel>
NodeLabelledMultiDiGraph<NodeLabel>::NodeLabelledMultiDiGraph(
  cow_ptr_t<typename NodeLabelledMultiDiGraph<NodeLabel>::Interface> ptr)
  : ptr(std::move(ptr)) {}

} // namespace FlexFlow

#endif
