#include "utils/relation/hemiunique_binrel_transform_l.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using L2 = ordered_value_type<1>;
using R = ordered_value_type<2>;
using F = std::function<L2(L const &)>;

template HemiuniqueBinaryRelation<L2, R>
    hemiunique_binrel_transform_l(HemiuniqueBinaryRelation<L, R> const &, F &&);

} // namespace FlexFlow
