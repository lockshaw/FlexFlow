#include "pcg/machine_space_coordinate.h"

namespace FlexFlow {

MachineSpaceCoordinate
  make_machine_space_1d_coordinate(nonnegative_int idx) {

  return MachineSpaceCoordinate{
    MachineSpace1dCoordinate{
      idx,
    },
  };
}

MachineSpaceCoordinate
  make_machine_space_2d_coordinate(nonnegative_int node_idx, nonnegative_int device_idx) {

  return MachineSpaceCoordinate{
    MachineSpace2dCoordinate{
      /*node_idx=*/node_idx,
      /*device_idx=*/device_idx,
    },
  };
}


} // namespace FlexFlow
