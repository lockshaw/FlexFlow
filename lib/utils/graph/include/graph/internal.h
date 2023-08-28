#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_INTERNAL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_INTERNAL_H

#include "labelled/labelled_downward_open.fwd.h"
#include "labelled/labelled_downward_open.fwd.h"
#include "labelled/labelled_upward_open.fwd.h"
#include "labelled/node_labelled.fwd.h"
#include "labelled/labelled_open.fwd.h"
#include "labelled/labelled_open_interfaces.fwd.h"
#include "labelled/labelled_upward_open_interfaces.fwd.h"
#include "labelled/labelled_downward_open_interfaces.fwd.h"
#include "labelled/node_labelled_interfaces.fwd.h"
#include "utils/graph/digraph.h"
#include "utils/graph/digraph_interfaces.h"
#include "utils/graph/multidigraph.h"
#include "utils/graph/multidigraph_interfaces.h"
#include "utils/graph/node.h"
#include "utils/graph/open_graph_interfaces.h"
#include "utils/graph/open_graphs.h"
#include "utils/graph/undirected.h"

namespace FlexFlow {

struct GraphInternal {
private:
  static OpenMultiDiGraph create_open_multidigraph(std::shared_ptr<IOpenMultiDiGraph>);
  static OpenMultiDiGraphView create_open_multidigraph_view(std::shared_ptr<IOpenMultiDiGraphView const>);

  static MultiDiGraph create_multidigraph(std::shared_ptr<IMultiDiGraph>);
  static MultiDiGraphView
      create_multidigraphview(std::shared_ptr<IMultiDiGraphView const>);

  static DiGraph create_digraph(std::shared_ptr<IDiGraph>);
  static DiGraphView create_digraphview(std::shared_ptr<IDiGraphView const>);

  static UndirectedGraph
      create_undirectedgraph(std::shared_ptr<IUndirectedGraph>);
  static UndirectedGraphView
      create_undirectedgraphview(std::shared_ptr<IUndirectedGraphView const>);

  static Graph create_graph(std::shared_ptr<IGraph>);
  static GraphView create_graphview(std::shared_ptr<IGraphView const>);

  template <typename NodeLabel>
  static NodeLabelledMultiDiGraphView<NodeLabel> create_node_labelled_multidigraph_view(
    std::shared_ptr<INodeLabelledMultiDiGraphView<NodeLabel> const>);

  template <typename N, typename E, typename I, typename O>
  static LabelledOpenMultiDiGraphView<N, E, I, O> create_labelled_open_multidigraph_view(
    std::shared_ptr<ILabelledOpenMultiDiGraphView<N, E, I, O> const>);

  template <typename NodeLabel>
  static NodeLabelledMultiDiGraphView<NodeLabel> create_node_labelled_open_multidigraph_view(
    std::shared_ptr<INodeLabelledMultiDiGraphView<NodeLabel> const>);

  template <typename N, typename E, typename I>
  static LabelledUpwardOpenMultiDiGraphView<N, E, I> create_labelled_upward_open_multidigraph_view(
    std::shared_ptr<ILabelledUpwardOpenMultiDiGraphView<N, E, I> const>);

  template <typename N, typename E, typename O>
  static LabelledDownwardOpenMultiDiGraphView<N, E, O> create_labelled_downward_open_multidigraph_view(
    std::shared_ptr<ILabelledDownwardOpenMultiDiGraphView<N, E, O> const>);

  friend struct MultiDiGraph;
  friend struct MultiDiGraphView;
  friend struct DiGraph;
  friend struct DiGraphView;
  friend struct UndirectedGraph;
  friend struct UndirectedGraphView;
  friend struct Graph;
  friend struct GraphView;
  friend struct OpenMultiDiGraphView;
  friend struct OpenMultiDiGraph;

  template <typename N, typename E, typename I, typename O>
  friend struct LabelledOpenMultiDiGraphView;

  template <typename N, typename E, typename I, typename O>
  friend struct LabelledOpenMultiDiGraph;

  template <typename NodeLabel>
  friend struct NodeLabelledMultiDiGraphView;

  template <typename NodeLabel>
  friend struct NodeLabelledMultiDiGraph;

  template <typename N, typename E, typename I>
  friend struct LabelledUpwardOpenMultiDiGraphView;

  template <typename N, typename E, typename I>
  friend struct LabelledUpwardOpenMultiDiGraph;

  template <typename N, typename E, typename O>
  friend struct LabelledDownwardOpenMultiDiGraphView;

  template <typename N, typename E, typename O>
  friend struct LabelledDownwardOpenMultiDiGraph;
};

} // namespace FlexFlow

#endif
