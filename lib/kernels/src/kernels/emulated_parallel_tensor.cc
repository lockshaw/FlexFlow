#include "kernels/emulated_parallel_tensor.h"

namespace FlexFlow {

bool emulated_parallel_tensors_are_equal(EmulatedParallelTensor const &lhs,
                                         EmulatedParallelTensor const &rhs)
{
  std::set<ParallelTensorSpaceCoordinate> lhs_coords = keys(lhs.shards);
  std::set<ParallelTensorSpaceCoordinate> rhs_coords = keys(rhs.shards);

  if (lhs_coords != rhs_coords) {
    return false;
  }

  return all_of(
    require_same(lhs_coords, rhs_coords),
    [&](ParallelTensorSpaceCoordinate const &c) -> bool {
      return accessors_are_equal(lhs.shards.at(c), rhs.shards.at(c));
    });
}

} // namespace FlexFlow
