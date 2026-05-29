#include "pcg/flat_machine_space_coordinate.h"
#include "pcg/machine_compute_resource_slice.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

FlatMachineSpaceCoordinate convert_machine_space_coord_to_flat(
    MachineSpaceCoordinate const &coord,
    MachineComputeResourceSlice const &machine_space) {
  return FlatMachineSpaceCoordinate{
      coord.node_idx * machine_space.num_gpus_per_node + coord.device_idx,
  };
}

MachineSpaceCoordinate resolve_flat_machine_space_coord(
    FlatMachineSpaceCoordinate const &coord,
    MachineComputeResourceSlice const &machine_space) {
  nonnegative_int node_idx = coord.idx / machine_space.num_gpus_per_node;
  nonnegative_int device_idx = coord.idx % machine_space.num_gpus_per_node;

  MachineSpaceCoordinate result = MachineSpaceCoordinate{
      /*node_idx=*/node_idx,
      /*device_idx=*/device_idx,
  };

  ASSERT(is_valid_machine_space_coordinate_in_slice(machine_space, result));

  return result;
}

} // namespace FlexFlow
