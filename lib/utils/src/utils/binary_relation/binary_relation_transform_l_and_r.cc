#include "utils/binary_relation/binary_relation_transform_l_and_r.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;
using L2 = ordered_value_type<2>;
using R2 = ordered_value_type<3>;
using FL = std::function<L2(L const &)>;
using FR = std::function<R2(R const &)>;

template BinaryRelation<L2, R2> binary_relation_transform_l_and_r(
    BinaryRelation<L, R> const &, FL &&, FR &&);

} // namespace FlexFlow
