#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_INTERFACES_FWD_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_UPWARD_OPEN_INTERFACES_FWD_H

namespace FlexFlow {

template <typename NodeLabel,
          typename EdgeLabel,
          typename OutputLabel = EdgeLabel>
struct ILabelledUpwardOpenMultiDiGraphView;

template <typename NodeLabel,
          typename EdgeLabel,
          typename OutputLabel = EdgeLabel>
struct ILabelledUpwardOpenMultiDiGraph;

}

#endif
