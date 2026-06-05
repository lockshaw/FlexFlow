#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FLAT_MACHINE_SPACE_OFFSET_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FLAT_MACHINE_SPACE_OFFSET_H

#include "pcg/flat_machine_space_coordinate.dtg.h"
#include "pcg/flat_machine_space_offset.dtg.h"

namespace FlexFlow {

FlatMachineSpaceCoordinate
    offset_flat_machine_space_coord_by(FlatMachineSpaceCoordinate const &coord,
                                       FlatMachineSpaceOffset const &offset);

} // namespace FlexFlow

#endif
