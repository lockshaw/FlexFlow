#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_TRANSFORM_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_TRANSFORM_KEYS_H

#include "utils/containers/transform.h"
#include "utils/many_to_one/many_to_one_from_unstructured_relation.h"
#include "utils/many_to_one/unstructured_relation_from_many_to_one.h"

namespace FlexFlow {

template <typename L1,
          typename R,
          typename F,
          typename L2 = std::invoke_result_t<F, L1>>
ManyToOne<L2, R> many_to_one_transform_keys(ManyToOne<L1, R> const &input,
                                            F &&f) {
  return many_to_one_from_unstructured_relation(
      transform(unstructured_relation_from_many_to_one(input),
                [&](std::pair<L1, R> const &p) -> std::pair<L2, R> {
                  return {f(p.first), p.second};
                }));
}

} // namespace FlexFlow

#endif
