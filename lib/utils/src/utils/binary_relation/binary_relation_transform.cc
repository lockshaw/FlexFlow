#include "utils/binary_relation/binary_relation_transform.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;
using L2 = ordered_value_type<2>;
using R2 = ordered_value_type<3>;
using F = std::function<std::pair<L2, R2>(L const &, R const &)>;

template BinaryRelation<L2, R2>
    binary_relation_transform(BinaryRelation<L, R> const &, F &&);

} // namespace FlexFlow
