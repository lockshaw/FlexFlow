#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_BINREL_TRANSFORM_L_AND_R_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_BINREL_TRANSFORM_L_AND_R_H

#include "utils/relation/hemiunique_binrel_transform_l.h"
#include "utils/relation/hemiunique_binrel_transform_r.h"

namespace FlexFlow {

template <typename L1,
          typename R1,
          typename FL,
          typename FR,
          typename L2 = std::invoke_result_t<FL, L1>,
          typename R2 = std::invoke_result_t<FR, R1>>
HemiuniqueBinaryRelation<L2, R2>
  hemiunique_binrel_transform_l_and_r(HemiuniqueBinaryRelation<L1, R1> const &r,
                                      FL &&fl,
                                      FR &&fr) {
  return hemiunique_binrel_transform_l(
    hemiunique_binrel_transform_r(
      r,
      fr),
    fl);
}

} // namespace FlexFlow

#endif
