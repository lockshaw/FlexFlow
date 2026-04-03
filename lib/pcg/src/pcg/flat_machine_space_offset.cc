#include "pcg/flat_machine_space_offset.h"
#include <libassert/assert.hpp>
#include "pcg/machine_compute_resource_slice.h"

namespace FlexFlow {

FlatMachineSpaceCoordinate
  offset_flat_machine_space_coord_by(FlatMachineSpaceCoordinate const &coord,
                                     FlatMachineSpaceOffset const &offset)
{
  return FlatMachineSpaceCoordinate{
    nonnegative_int{coord.idx.unwrap_nonnegative() + offset.offset},
  };
}

} // namespace FlexFlow
