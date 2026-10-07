#include "utils/benchmark_utils/random_vector.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using F = std::function<T(std::mt19937 &)>;

template std::vector<T> random_vector(std::mt19937 &, nonnegative_int, F &&);

} // namespace FlexFlow
