#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_SPACE_1D_OFFSET_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_SPACE_1D_OFFSET_H

#include "pcg/machine_space_1d_coordinate.dtg.h"
#include "pcg/machine_space_1d_offset.dtg.h"

namespace FlexFlow {

MachineSpace1dOffset get_machine_space_1d_offset_from_coordinate(
    MachineSpace1dCoordinate const &start, MachineSpace1dCoordinate const &coord);

MachineSpace1dCoordinate offset_machine_space_1d_coordinate_by(
    MachineSpace1dCoordinate const &start, MachineSpace1dOffset const &offset);


} // namespace FlexFlow

#endif
