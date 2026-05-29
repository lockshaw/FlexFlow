#include "pcg/unresolved_machine_space_offset.h"
#include "pcg/flat_machine_space_coordinate.h"
#include "pcg/flat_machine_space_offset.h"
#include "pcg/machine_space_offset.h"
#include "utils/overload.h"

namespace FlexFlow {

MachineSpaceCoordinate offset_machine_space_coordinate_by_unresolved(
    MachineComputeResourceSlice const &machine_space,
    MachineSpaceCoordinate const &coord,
    UnresolvedMachineSpaceOffset const &offset) {
  return offset.visit<MachineSpaceCoordinate>(overload{
      [&](MachineSpaceOffset const &standard_offset) -> MachineSpaceCoordinate {
        return offset_machine_space_coordinate_by(coord, standard_offset);
      },
      [&](FlatMachineSpaceOffset const &flat_offset) -> MachineSpaceCoordinate {
        return resolve_flat_machine_space_coord(
            offset_flat_machine_space_coord_by(
                convert_machine_space_coord_to_flat(coord, machine_space),
                flat_offset),
            machine_space);
      }});
}

} // namespace FlexFlow
