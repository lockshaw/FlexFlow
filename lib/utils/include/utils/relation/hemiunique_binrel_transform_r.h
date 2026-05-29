#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_BINREL_TRANSFORM_R_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_BINREL_TRANSFORM_R_H

#include "utils/one_to_many/one_to_many_transform_values.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/relation/hemiunique_binary_relation.h"
#include "utils/many_to_one/many_to_one_transform_values.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename L,
          typename R1,
          typename F,
          typename R2 = std::invoke_result_t<F, R1>>
HemiuniqueBinaryRelation<L, R2>
  hemiunique_binrel_transform_r(HemiuniqueBinaryRelation<L, R1> const &r,
                                F &&f) {
  return r.template visit<HemiuniqueBinaryRelation<L, R2>>(overload {
    [&](bidict<L, R1> const &b)
      -> HemiuniqueBinaryRelation<L, R2>
    {
      return HemiuniqueBinaryRelation<L, R2>{
        transform_values(b, f),
      };
    },
    [&](OneToMany<L, R1> const &otm)
      -> HemiuniqueBinaryRelation<L, R2>
    {
      return HemiuniqueBinaryRelation<L, R2>{
        one_to_many_transform_values(otm, f),
      };
    },
    [&](ManyToOne<L, R1> const &mto)
      -> HemiuniqueBinaryRelation<L, R2>
    {
      return HemiuniqueBinaryRelation<L, R2>{
        many_to_one_transform_values(mto, f),
      };
    },
  });
}

} // namespace FlexFlow

#endif
