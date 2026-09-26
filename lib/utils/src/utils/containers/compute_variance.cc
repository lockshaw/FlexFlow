#include "utils/containers/compute_variance.h"
#include "utils/containers/compute_mean.h"
#include <libassert/assert.hpp>
#include "utils/containers/transform.h"

namespace FlexFlow {

float compute_variance(std::vector<float> const &vs) {
  ASSERT(!vs.empty());

  float mean = compute_mean(vs);

  return compute_mean(
    transform(
      vs,
      [&](float x) -> float {
        float dev = x - mean;
        return dev * dev;
      }));
}

} // namespace FlexFlow
