#include "task-spec/dynamic_graph/dynamic_node_slot.h"

namespace FlexFlow {

DynamicValueAttrs dynamic_value_attrs_for_node_slot(DynamicNodeSlot const &slot) {
  switch (slot.direction) {
    case TensorDirection::INCOMING:
      return slot.inputs.at(slot.slot_name);
    case TensorDirection::OUTPUT:
      return slot.outputs.at(slot.slot_name);
    default:
      PANIC("Unexpected direction {}", slot.direction);
  }
}


} // namespace FlexFlow
