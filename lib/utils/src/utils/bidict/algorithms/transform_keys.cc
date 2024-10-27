#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;
using F = std::function<K2(K)>;

template
  bidict<K2, V> transform_keys(bidict<K, V> const &, F &&);

} // namespace FlexFlow
