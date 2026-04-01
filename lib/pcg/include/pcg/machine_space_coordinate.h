#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_SPACE_COORDINATE_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_SPACE_COORDINATE_H

#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

MachineSpaceCoordinate
  make_machine_space_1d_coordinate(nonnegative_int idx);

MachineSpaceCoordinate
  make_machine_space_2d_coordinate(nonnegative_int node_idx, nonnegative_int device_idx);

} // namespace FlexFlow

#endif
