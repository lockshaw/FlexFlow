#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FLAT_MACHINE_SPACE_COORDINATE_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FLAT_MACHINE_SPACE_COORDINATE_H

#include "pcg/flat_machine_space_coordinate.dtg.h"
#include "pcg/machine_compute_resource_slice.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

FlatMachineSpaceCoordinate convert_machine_space_coord_to_flat(
    MachineSpaceCoordinate const &coord,
    MachineComputeResourceSlice const &machine_space);

MachineSpaceCoordinate resolve_flat_machine_space_coord(
    FlatMachineSpaceCoordinate const &coord,
    MachineComputeResourceSlice const &machine_space);

} // namespace FlexFlow

#endif
