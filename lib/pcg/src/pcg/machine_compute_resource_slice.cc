#include "pcg/machine_compute_resource_slice.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

MachineComputeResourceSlice
    compute_slice_from_specification(MachineComputeSpecification const &spec) {

  return MachineComputeResourceSlice{
      /*num_nodes=*/spec.num_nodes,
      /*num_gpus_per_node=*/spec.num_gpus_per_node,
  };
}

positive_int
    get_total_num_devices_in_slice(MachineComputeResourceSlice const &slice) {
  return slice.num_nodes * slice.num_gpus_per_node;
}

bool is_valid_machine_space_coordinate_in_slice(
    MachineComputeResourceSlice const &slice,
    MachineSpaceCoordinate const &coord) {
  return coord.visit<bool>(overload {
    [&](MachineSpace1dCoordinate const &c) -> bool {
      return (c.idx < get_total_num_devices_in_slice(slice));
    },
    [&](MachineSpace2dCoordinate const &c) -> bool {
      return (c.node_idx < slice.num_nodes) &&
        (c.device_idx < slice.num_gpus_per_node);
    }
  });
}

} // namespace FlexFlow
