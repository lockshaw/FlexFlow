#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_TRANSFORM_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_TRANSFORM_VALUES_H

#include "utils/containers/transform.h"
#include "utils/many_to_one/many_to_one_from_unstructured_relation.h"
#include "utils/many_to_one/unstructured_relation_from_many_to_one.h"

namespace FlexFlow {

template <typename L,
          typename R1,
          typename F,
          typename R2 = std::invoke_result_t<F, R1>>
ManyToOne<L, R2> many_to_one_transform_values(ManyToOne<L, R1> const &input,
                                              F &&f) {
  return many_to_one_from_unstructured_relation(
      transform(unstructured_relation_from_many_to_one(input),
                [&](std::pair<L, R1> const &p) -> std::pair<L, R2> {
                  return {p.first, f(p.second)};
                }));
}

} // namespace FlexFlow

#endif
