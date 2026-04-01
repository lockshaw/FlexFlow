#include "pcg/machine_space_offset.h"
#include "pcg/machine_space_1d_offset.h"
#include "pcg/machine_space_2d_offset.h"
#include "utils/exception.h"
#include "utils/overload.h"

namespace FlexFlow {

MachineSpaceOffset get_machine_space_offset_from_coordinate(
    MachineSpaceCoordinate const &start, MachineSpaceCoordinate const &coord) {

  ASSERT(start.is_1d() == coord.is_1d());

  return start.visit<MachineSpaceOffset>(overload {
    [&](MachineSpace1dCoordinate const &start_1d) -> MachineSpaceOffset {
      return MachineSpaceOffset{
        get_machine_space_1d_offset_from_coordinate(
          start_1d, coord.require_1d()),
      };
    },
    [&](MachineSpace2dCoordinate const &start_2d) -> MachineSpaceOffset {
      return MachineSpaceOffset{
        get_machine_space_2d_offset_from_coordinate(
          start_2d, coord.require_2d()),
      };
    },
  });
}

MachineSpaceCoordinate offset_machine_space_coordinate_by(
    MachineSpaceCoordinate const &start, MachineSpaceOffset const &offset)
{
  ASSERT(start.is_1d() == offset.is_1d());

  return start.visit<MachineSpaceCoordinate>(overload {
    [&](MachineSpace1dCoordinate const &start_1d) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
        offset_machine_space_1d_coordinate_by(
          start_1d, offset.require_1d()),
      };
    },
    [&](MachineSpace2dCoordinate const &start_2d) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
        offset_machine_space_2d_coordinate_by(
          start_2d, offset.require_2d()),
      };
    },
  });
}


} // namespace FlexFlow
