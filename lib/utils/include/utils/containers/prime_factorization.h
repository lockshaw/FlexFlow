#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PRIME_FACTORIZATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_PRIME_FACTORIZATION_H

#include "utils/int_ge_two/int_ge_two.h"

namespace FlexFlow {

std::multiset<int_ge_two>
  prime_factorization(positive_int x);

std::multiset<int_ge_two>
  prime_factorization(int_ge_two x);

} // namespace FlexFlow

#endif
