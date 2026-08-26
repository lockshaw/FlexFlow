#include "pcg/machine_space_offset.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

MachineSpaceOffset get_machine_space_offset_from_coordinate(
    MachineSpaceCoordinate const &start, MachineSpaceCoordinate const &coord) {
  ASSERT(start.device_idx <= coord.device_idx,
         "The start device_idx is greater than one of the coord device_idx."
         "Are you sure you didn't swap them?");

  ASSERT(start.node_idx <= coord.device_idx,
         "The start node_idx is greater than one of the coord node_idx."
         "Are you sure you didn't swap them?");

  return MachineSpaceOffset{
      /*node_offset=*/coord.node_idx.unwrap_nonnegative() -
          start.node_idx.unwrap_nonnegative(),
      /*device_offset=*/coord.device_idx.unwrap_nonnegative() -
          start.device_idx.unwrap_nonnegative(),
  };
}

MachineSpaceCoordinate
    offset_machine_space_coordinate_by(MachineSpaceCoordinate const &start,
                                       MachineSpaceOffset const &offset) {
  return MachineSpaceCoordinate{
      /*node_idx=*/nonnegative_int{
          start.node_idx.unwrap_nonnegative() + offset.node_offset,
      },
      /*device_idx=*/
      nonnegative_int{
          start.device_idx.unwrap_nonnegative() + offset.device_offset,
      },
  };
}

} // namespace FlexFlow
