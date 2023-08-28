#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_DECL_H

#include "labelled_upward_open_interfaces.h"
#include "labelled_upward_open.fwd.h"
#include "utils/test_types.h"

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
struct LabelledUpwardOpenMultiDiGraphView {
private:
  using Interface =
      ILabelledUpwardOpenMultiDiGraphView<NodeLabel, EdgeLabel, InputLabel>;

public:
  LabelledUpwardOpenMultiDiGraphView() = delete;

  template <typename OutputLabel>
  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const;

  InputLabel const &at(InputMultiDiEdge const &e) const;
  EdgeLabel const &at(MultiDiEdge const &e) const;
  NodeLabel const &at(Node const &n) const;

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const;
  std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &q) const;

  template <typename BaseImpl, typename... Args>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledUpwardOpenMultiDiGraphView>::type
      create(Args &&...args);

private:
  LabelledUpwardOpenMultiDiGraphView(std::shared_ptr<Interface const>);

private:
  std::shared_ptr<Interface const> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledUpwardOpenMultiDiGraphView<test_types::cmp,
                                       test_types::cmp,
                                       test_types::cmp>);
CHECK_NOT_ABSTRACT(LabelledUpwardOpenMultiDiGraphView<test_types::cmp,
                                                      test_types::cmp,
                                                      test_types::cmp>);

template <typename NodeLabel, typename EdgeLabel, typename InputLabel>
struct LabelledUpwardOpenMultiDiGraph {
private:
  using Interface =
      ILabelledUpwardOpenMultiDiGraph<NodeLabel, EdgeLabel, InputLabel>;

public:
  LabelledUpwardOpenMultiDiGraph() = delete;

  template <typename OutputLabel>
  operator LabelledOpenMultiDiGraphView<NodeLabel,
                                        EdgeLabel,
                                        InputLabel,
                                        OutputLabel>() const;

  InputLabel const &at(InputMultiDiEdge const &e) const;
  InputLabel &at(InputMultiDiEdge const &e);

  EdgeLabel const &at(MultiDiEdge const &e) const;
  EdgeLabel &at(MultiDiEdge const &e);

  NodeLabel const &at(Node const &n) const;
  NodeLabel &at(Node const &n);

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const;
  std::unordered_set<UpwardOpenMultiDiEdge>
      query_edges(UpwardOpenMultiDiEdgeQuery const &q) const;

  Node add_node(NodeLabel const &l);
  void add_node_unsafe(Node const &n, NodeLabel const &l);
  void add_edge(MultiDiEdge const &e, EdgeLabel const &l);
  void add_edge(InputMultiDiEdge const &e, InputLabel const &l);

  template <typename BaseImpl>
  static typename std::enable_if<std::is_base_of<Interface, BaseImpl>::value,
                                 LabelledUpwardOpenMultiDiGraph>::type
      create();

private:
  LabelledUpwardOpenMultiDiGraph(cow_ptr_t<Interface>);

private:
  cow_ptr_t<Interface> ptr;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(
    LabelledUpwardOpenMultiDiGraph<int, int, int>);

} // namespace FlexFlow

#endif
