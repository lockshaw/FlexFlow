#include "utils/relation/hemiunique_binrel_transform_r.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R1 = value_type<1>;
using R2 = value_type<2>;
using F = std::function<R2(R1 const &)>;

template
  HemiuniqueBinaryRelation<L, R2>
    hemiunique_binrel_transform_r(HemiuniqueBinaryRelation<L, R1> const &, F &&);

} // namespace FlexFlow
