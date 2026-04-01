#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_SPACE_2D_OFFSET_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_SPACE_2D_OFFSET_H

#include "pcg/machine_space_2d_coordinate.dtg.h"
#include "pcg/machine_space_2d_offset.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/machine_space_offset.dtg.h"

namespace FlexFlow {

MachineSpace2dOffset get_machine_space_2d_offset_from_coordinate(
    MachineSpace2dCoordinate const &start, MachineSpace2dCoordinate const &coord);

MachineSpace2dCoordinate offset_machine_space_2d_coordinate_by(
    MachineSpace2dCoordinate const &start, MachineSpace2dOffset const &offset);

} // namespace FlexFlow

#endif
