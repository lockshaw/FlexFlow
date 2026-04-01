#include "pcg/machine_space_1d_offset.h"

namespace FlexFlow {

MachineSpace1dOffset get_machine_space_1d_offset_from_coordinate(
    MachineSpace1dCoordinate const &start, MachineSpace1dCoordinate const &coord)
{
  return MachineSpace1dOffset{
    coord.idx.unwrap_nonnegative() - start.idx.unwrap_nonnegative(),
  };
}

MachineSpace1dCoordinate offset_machine_space_1d_coordinate_by(
    MachineSpace1dCoordinate const &start, MachineSpace1dOffset const &offset)
{
  return MachineSpace1dCoordinate{
    nonnegative_int{
      start.idx.unwrap_nonnegative() + offset.offset,
    },
  };
}


} // namespace FlexFlow
