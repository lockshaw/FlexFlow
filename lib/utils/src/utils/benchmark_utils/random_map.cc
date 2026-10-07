#include "utils/benchmark_utils/random_map.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;

using KF = std::function<K(std::mt19937 &)>;
using VF = std::function<V(std::mt19937 &)>;

template std::map<K, V>
    random_map(std::mt19937 &, nonnegative_int, KF &&, VF &&);

} // namespace FlexFlow
