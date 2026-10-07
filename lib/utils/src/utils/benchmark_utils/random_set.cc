#include "utils/benchmark_utils/random_set.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;
using F = std::function<T(std::mt19937 &)>;

template std::set<T> random_set(std::mt19937 &, nonnegative_int, F &&);

} // namespace FlexFlow
