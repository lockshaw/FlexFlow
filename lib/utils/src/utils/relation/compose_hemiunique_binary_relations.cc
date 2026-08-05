#include "utils/relation/compose_hemiunique_binary_relations.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using C = ordered_value_type<1>;
using R = ordered_value_type<2>;

template
  HemiuniqueBinaryRelation<L, R> compose_hemiunique_binary_relations(
      HemiuniqueBinaryRelation<L, C> const &,
      HemiuniqueBinaryRelation<C, R> const &);

} // namespace FlexFlow
