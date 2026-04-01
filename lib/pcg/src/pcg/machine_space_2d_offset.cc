#include "pcg/machine_space_2d_offset.h"

namespace FlexFlow {

MachineSpace2dOffset get_machine_space_offset_from_coordinate(
    MachineSpace2dCoordinate const &start, MachineSpace2dCoordinate const &coord)
{
  ASSERT(start.device_idx <= coord.device_idx,
         "The start device_idx is greater than one of the coord device_idx."
         "Are you sure you didn't swap them?");

  ASSERT(start.node_idx <= coord.device_idx,
         "The start node_idx is greater than one of the coord node_idx."
         "Are you sure you didn't swap them?");

  return MachineSpace2dOffset{
      /*node_offset=*/coord.node_idx.unwrap_nonnegative() -
          start.node_idx.unwrap_nonnegative(),
      /*device_offset=*/coord.device_idx.unwrap_nonnegative() -
          start.device_idx.unwrap_nonnegative(),
  };
}

MachineSpace2dCoordinate offset_machine_space_coordinate_by(
    MachineSpace2dCoordinate const &start, MachineSpace2dOffset const &offset)
{
  return MachineSpace2dCoordinate{
    /*node_idx=*/nonnegative_int{
      start.node_idx.unwrap_nonnegative() + offset.node_offset,
    },
    /*device_idx=*/nonnegative_int{
      start.node_idx.unwrap_nonnegative() + offset.device_offset,
    },
  };
}


} // namespace FlexFlow
