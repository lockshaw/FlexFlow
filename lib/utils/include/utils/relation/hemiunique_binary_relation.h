#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_RELATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_RELATION_H

#include "utils/relation/hemiunique_binary_relation.dtg.h"
#include "utils/overload.h"
#include "utils/relation/uniqueness.dtg.h"
#include "utils/one_to_many/invert_one_to_many.h"
#include "utils/many_to_one/invert_many_to_one.h"

namespace FlexFlow {

template <typename L, typename R>
Uniqueness hemiunique_binary_relation_get_uniqueness(HemiuniqueBinaryRelation<L, R> const &r)
{
  return r.template visit<Uniqueness>(overload {
    [](bidict<L, R> const &) -> Uniqueness {
      return Uniqueness::BIUNIQUE;  
    },
    [](OneToMany<L, R> const &) -> Uniqueness {
      return Uniqueness::LEFT_UNIQUE;
    },
    [](ManyToOne<L, R> const &) -> Uniqueness {
      return Uniqueness::RIGHT_UNIQUE; 
    },
  });
}

template <typename L, typename R>
std::unordered_set<L> hemiunique_binrel_left_entries(HemiuniqueBinaryRelation<L, R> const &r) {
  return r.template visit<std::unordered_set<L>>(overload {
    [](bidict<L, R> const &b) -> std::unordered_set<L> {
      return b.left_values();
    },
    [](OneToMany<L, R> const &otm) -> std::unordered_set<L> {
      return otm.left_values(); 
    },
    [](ManyToOne<L, R> const &mto) -> std::unordered_set<L> {
      return mto.left_values(); 
    },
  });
}

template <typename L, typename R>
std::unordered_set<R> hemiunique_binrel_right_entries(HemiuniqueBinaryRelation<L, R> const &r) {
  return r.template visit<std::unordered_set<R>>(overload {
    [](bidict<L, R> const &b) -> std::unordered_set<R> {
      return b.right_values();
    },
    [](OneToMany<L, R> const &otm) -> std::unordered_set<R> {
      return otm.right_values(); 
    },
    [](ManyToOne<L, R> const &mto) -> std::unordered_set<R> {
      return mto.right_values(); 
    },
  });
}

template <typename L, typename R>
HemiuniqueBinaryRelation<R, L> invert_hemiunique_binary_relation(HemiuniqueBinaryRelation<L, R> const &r) {
  return r.template visit<HemiuniqueBinaryRelation<R, L>>(overload {
    [](bidict<L, R> const &b) -> HemiuniqueBinaryRelation<R, L> {
      return HemiuniqueBinaryRelation<R, L>{
        b.reversed(),
      };
    },
    [](OneToMany<L, R> const &otm) -> HemiuniqueBinaryRelation<R, L> {
      return HemiuniqueBinaryRelation<R, L>{
        invert_one_to_many(otm),
      };
    },
    [](ManyToOne<L, R> const &mto) -> HemiuniqueBinaryRelation<R, L> {
      return HemiuniqueBinaryRelation<R, L>{
        invert_many_to_one(mto),
      };
    },
  });
}

} // namespace FlexFlow

#endif
