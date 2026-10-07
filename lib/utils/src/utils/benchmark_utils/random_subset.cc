#include "utils/benchmark_utils/random_subset.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template std::set<T>
    random_subset(std::mt19937 &, nonnegative_int, std::set<T> const &);

} // namespace FlexFlow
