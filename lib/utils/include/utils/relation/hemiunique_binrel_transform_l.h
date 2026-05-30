#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_BINREL_TRANSFORM_L_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_BINREL_TRANSFORM_L_H

#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/many_to_one/many_to_one_transform_keys.h"
#include "utils/one_to_many/one_to_many_transform_keys.h"
#include "utils/overload.h"
#include "utils/relation/hemiunique_binary_relation.h"

namespace FlexFlow {

template <typename L,
          typename R,
          typename F,
          typename L2 = std::invoke_result_t<F, L>>
HemiuniqueBinaryRelation<L2, R>
    hemiunique_binrel_transform_l(HemiuniqueBinaryRelation<L, R> const &r,
                                  F &&f) {
  return r.template visit<HemiuniqueBinaryRelation<L2, R>>(overload{
      [&](bidict<L, R> const &b) -> HemiuniqueBinaryRelation<L2, R> {
        return HemiuniqueBinaryRelation<L2, R>{
            transform_keys(b, f),
        };
      },
      [&](OneToMany<L, R> const &otm) -> HemiuniqueBinaryRelation<L2, R> {
        return HemiuniqueBinaryRelation<L2, R>{
            one_to_many_transform_keys(otm, f),
        };
      },
      [&](ManyToOne<L, R> const &mto) -> HemiuniqueBinaryRelation<L2, R> {
        return HemiuniqueBinaryRelation<L2, R>{
            many_to_one_transform_keys(mto, f),
        };
      },
  });
}

} // namespace FlexFlow

#endif
