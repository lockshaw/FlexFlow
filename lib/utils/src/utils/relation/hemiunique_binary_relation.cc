#include "utils/relation/hemiunique_binary_relation.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  Uniqueness hemiunique_binary_relation_get_uniqueness(HemiuniqueBinaryRelation<L, R> const &);

template
  std::unordered_set<L> hemiunique_binrel_left_entries(HemiuniqueBinaryRelation<L, R> const &);

template
  std::unordered_set<R> hemiunique_binrel_right_entries(HemiuniqueBinaryRelation<L, R> const &);

template
  HemiuniqueBinaryRelation<R, L> invert_hemiunique_binary_relation(HemiuniqueBinaryRelation<L, R> const &);

} // namespace FlexFlow
