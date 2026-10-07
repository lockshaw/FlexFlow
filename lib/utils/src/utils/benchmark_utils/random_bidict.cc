#include "utils/benchmark_utils/random_bidict.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;
using FL = std::function<L(std::mt19937 &)>;
using FR = std::function<R(std::mt19937 &)>;

template bidict<L, R>
    random_bidict(std::mt19937 &, nonnegative_int, FL &&, FR &&);

} // namespace FlexFlow
