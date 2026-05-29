#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_COMPOSE_HEMIUNIQUE_BINARY_RELATIONS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_COMPOSE_HEMIUNIQUE_BINARY_RELATIONS_H

#include "utils/one_to_many/one_to_many_from_bidict.h"
namespace FlexFlow {

template <typename L, typename C, typename R>
HemiuniqueBinaryRelation<L, R> compose_hemiunique_binary_relations(
    HemiuniqueBinaryRelation<L, C> const &l_rel,
    HemiuniqueBinaryRelation<C, R> const &r_rel) {
  Uniqueness l_uniqueness = l_rel.get_uniqueness();
  Uniqueness r_uniqueness = r_rel.get_uniqueness();

  auto is = [&](Uniqueness l, Uniqueness r) -> bool {
    return l == l_uniqueness && r == r_uniqueness;
  };

  if (is(Uniqueness::LEFT_UNIQUE, Uniqueness::BIUNIQUE)) {
    return HemiuniqueBinaryRelation<L, R>{
        compose_one_to_manys(l_rel.require_strictly_left_unique(),
                             one_to_many_from_bidict(r_rel.require_biunique())),
    };
  } else if (is(Uniqueness::BIUNIQUE, Uniqueness::LEFT_UNIQUE)) {
    return HemiuniqueBinaryRelation<L, R>{
        compose_one_to_manys(one_to_many_from_bidict(l_rel.require_biunique()),
                             r_rel.require_strictly_left_unique()),
    };
  } else if (is(Uniqueness::RIGHT_UNIQUE, Uniqueness::BIUNIQUE)) {
    return HemiuniqueBinaryRelation<L, R>{
        compose_many_to_ones(l_rel.require_strictly_right_unique(),
                             many_to_one_from_bidict(r_rel.require_biunique())),
    };
  } else if (is(Uniqueness::BIUNIQUE, Uniqueness::RIGHT_UNIQUE)) {
    return HemiuniqueBinaryRelation<L, R>{
        compose_many_to_ones(many_to_one_from_bidict(r_rel.require_biunique()),
                             r_rel.require_strictly_right_unique()),
    };
  }
}

} // namespace FlexFlow

#endif
