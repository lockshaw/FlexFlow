#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_IS_BIUNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_IS_BIUNIQUE_H

#include "utils/one_to_many/one_to_many.h"
#include "utils/containers/all_of.h"
#include "utils/containers/values.h"

namespace FlexFlow {

template <typename L, typename R>
bool one_to_many_is_biunique(OneToMany<L, R> const &otm) {
  return all_of(values(otm.l_to_r()),
                [](nonempty_unordered_set<R> const &rs) {
                  return rs.num_elements() == 1_p;
                });
}

} // namespace FlexFlow

#endif
