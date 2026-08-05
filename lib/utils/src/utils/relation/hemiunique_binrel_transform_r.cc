#include "utils/relation/hemiunique_binrel_transform_r.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R1 = ordered_value_type<1>;
using R2 = ordered_value_type<2>;
using F = std::function<R2(R1 const &)>;

template HemiuniqueBinaryRelation<L, R2>
    hemiunique_binrel_transform_r(HemiuniqueBinaryRelation<L, R1> const &,
                                  F &&);

} // namespace FlexFlow
