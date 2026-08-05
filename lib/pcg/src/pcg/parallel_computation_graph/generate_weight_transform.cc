#include "pcg/parallel_computation_graph/generate_weight_transform.h"
#include "op-attrs/ff_ordered/ff_ordered_enumerate.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::set<ParallelOpAttrs>
    generate_weight_transform(ParallelTensorDimDegrees const &goal) {
  std::set<ParallelOpAttrs> result;

  positive_int sum_degree = goal.sum_degree.value;
  ASSERT(sum_degree == 1,
         "generate_weight_transform currently only supports sum_degree = 1");

  positive_int discard_copy_degree = goal.discard_copy_degree.value;
  if (discard_copy_degree != 1) {
    result.insert(
        ParallelOpAttrs{ReplicateAttrs{int_ge_two{discard_copy_degree}}});
  }

  for (auto const &[shard_dim, shard_degree] : ff_ordered_enumerate(goal.shard_degrees)) {
    if (shard_degree != 1) {
      result.insert(ParallelOpAttrs{
          RepartitionAttrs{shard_dim, int_ge_two{shard_degree}}});
    }
  }

  return result;
}

} // namespace FlexFlow
