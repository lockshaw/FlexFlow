#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_DECL
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_NODE_LABELLED_DECL

#include "node_labelled_interfaces.h"
#include "utils/subtyping.h"

namespace FlexFlow {

template <typename NodeLabel>
struct NodeLabelledMultiDiGraphView {
private:
  using Interface = INodeLabelledMultiDiGraphView<NodeLabel>;

public:
  NodeLabelledMultiDiGraphView() = delete;
  NodeLabelledMultiDiGraphView(NodeLabelledMultiDiGraphView const &) = default;
  NodeLabelledMultiDiGraphView &
      operator=(NodeLabelledMultiDiGraphView const &) = default;

  operator MultiDiGraphView() const;

  NodeLabel const &at(Node const &n) const;

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const;

  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const;

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledMultiDiGraphView>::type
      create(Args &&...args);

private:
  std::shared_ptr<Interface const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(NodeLabelledMultiDiGraphView<int>);
MAKE_SUBTYPING_RELATION(graph_subtype, NodeLabelledMultiDiGraphView, 1, MultiDiGraphView);

template <typename NodeLabel>
struct NodeLabelledMultiDiGraph {
private:
  using Interface = INodeLabelledMultiDiGraph<NodeLabel>;

public:
  NodeLabelledMultiDiGraph() = delete;
  NodeLabelledMultiDiGraph(NodeLabelledMultiDiGraph const &) = default;
  NodeLabelledMultiDiGraph &
      operator=(NodeLabelledMultiDiGraph const &) = default;

  operator NodeLabelledMultiDiGraphView<NodeLabel>() const;

  friend void swap(NodeLabelledMultiDiGraph &lhs,
                   NodeLabelledMultiDiGraph &rhs);

  Node add_node(NodeLabel const &l);
  NodeLabel &at(Node const &n);
  NodeLabel const &at(Node const &n) const;

  void add_edge(MultiDiEdge const &e);

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const;
  std::unordered_set<MultiDiEdge> query_edges(MultiDiEdgeQuery const &q) const;

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 NodeLabelledMultiDiGraph>::type
      create();

private:
  NodeLabelledMultiDiGraph(cow_ptr_t<Interface>);

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(NodeLabelledMultiDiGraph<int>);
MAKE_SUBTYPING_RELATION(graph_subtype, NodeLabelledMultiDiGraph, 1, NodeLabelledMultiDiGraphView, 1);

} // namespace FlexFlow

#endif
