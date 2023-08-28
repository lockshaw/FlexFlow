#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_OPEN_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_OPEN_DECL_H

#include "labelled_downward_open_interfaces.h"
#include "labelled_open.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
struct LabelledDownwardOpenMultiDiGraphView {
private:
  using Interface =
      ILabelledDownwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, OutputLabel>;

public:
  template <typename InputLabel>
  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const;

  OutputLabel const &at(OutputMultiDiEdge const &e) const;
  EdgeLabel const &at(MultiDiEdge const &e) const;
  NodeLabel const &at(Node const &n) const;

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const;
  std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &q) const;

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledDownwardOpenMultiDiGraphView>::type
      create(Args &&...args);

private:
  std::shared_ptr<Interface const> ptr;
};

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel>
struct LabelledDownardOpenMultiDiGraph {
private:
  using Interface =
      ILabelledDownwardOpenMultiDiGraph<NodeLabel, EdgeLabel, OutputLabel>;

public:
  OutputLabel const &at(OutputMultiDiEdge const &e) const;
  OutputLabel &at(OutputMultiDiEdge const &e);

  EdgeLabel const &at(MultiDiEdge const &e) const;
  EdgeLabel &at(MultiDiEdge const &e);

  NodeLabel const &at(Node const &n) const;
  NodeLabel &at(Node const &n);

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const;
  std::unordered_set<DownwardOpenMultiDiEdge>
      query_edges(DownwardOpenMultiDiEdgeQuery const &q) const;

  Node add_node(NodeLabel const &l);
  void add_node_unsafe(Node const &n);
  void remove_node_unsafe(Node const &n);

  void add_edge(MultiDiEdge const &e, EdgeLabel const &l);
  void add_edge(OutputMultiDiEdge const &e, OutputLabel const &l);
  void remove_edge(OutputMultiDiEdge const &e);
  void remove_edge(MultiDiEdge const &e);

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledDownardOpenMultiDiGraph>::type
      create();

private:
  cow_ptr_t<Interface> ptr;
};

} // namespace FlexFlow

#endif
