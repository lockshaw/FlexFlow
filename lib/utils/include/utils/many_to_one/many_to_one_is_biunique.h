#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_IS_BIUNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_IS_BIUNIQUE_H

#include "utils/containers/all_of.h"
#include "utils/containers/values.h"
#include "utils/many_to_one/many_to_one.h"

namespace FlexFlow {

template <typename L, typename R>
bool many_to_one_is_biunique(ManyToOne<L, R> const &mto) {
  return all_of(values(mto.r_to_l()), [](nonempty_unordered_set<L> const &ls) {
    return ls.num_elements() == 1_p;
  });
}

} // namespace FlexFlow

#endif
