#include "utils/relation/hemiunique_binrel_transform_l_and_r.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L1 = ordered_value_type<0>;
using R1 = ordered_value_type<1>;
using L2 = ordered_value_type<2>;
using R2 = ordered_value_type<3>;
using FL = std::function<L2(L1 const &)>;
using FR = std::function<R2(R1 const &)>;

template HemiuniqueBinaryRelation<L2, R2> hemiunique_binrel_transform_l_and_r(
    HemiuniqueBinaryRelation<L1, R1> const &, FL &&, FR &&);

} // namespace FlexFlow
