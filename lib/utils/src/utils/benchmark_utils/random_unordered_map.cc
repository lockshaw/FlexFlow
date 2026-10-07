#include "utils/benchmark_utils/random_unordered_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

using KF = std::function<K(std::mt19937 &)>;
using VF = std::function<V(std::mt19937 &)>;

template std::unordered_map<K, V>
    random_unordered_map(std::mt19937 &, nonnegative_int, KF &&, VF &&);

} // namespace FlexFlow
