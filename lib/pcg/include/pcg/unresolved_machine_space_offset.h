#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_UNRESOLVED_MACHINE_SPACE_OFFSET_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_UNRESOLVED_MACHINE_SPACE_OFFSET_H

#include "pcg/machine_compute_resource_slice.dtg.h"
#include "pcg/unresolved_machine_space_offset.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

MachineSpaceCoordinate offset_machine_space_coordinate_by_unresolved(
  MachineComputeResourceSlice const &machine_space,
  MachineSpaceCoordinate const &coord,
  UnresolvedMachineSpaceOffset const &offset);

} // namespace FlexFlow

#endif
