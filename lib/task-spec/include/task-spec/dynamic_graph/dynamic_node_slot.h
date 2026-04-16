#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_SLOT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_SLOT_H

#include "task-spec/dynamic_graph/dynamic_node_slot.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"

namespace FlexFlow {

DynamicValueAttrs dynamic_value_attrs_for_node_slot(DynamicNodeSlot const &);

} // namespace FlexFlow

#endif
