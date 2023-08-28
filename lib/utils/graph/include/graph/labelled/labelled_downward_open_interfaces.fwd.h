#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_INTERFACES_FWD_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_LABELLED_DOWNWARD_INTERFACES_FWD_H

namespace FlexFlow {

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel = EdgeLabel>
struct ILabelledDownwardOpenMultiDiGraphView;

template <typename NodeLabel, typename EdgeLabel, typename OutputLabel = EdgeLabel>
struct ILabelledDownwardOpenMultiDiGraph;

}

#endif
