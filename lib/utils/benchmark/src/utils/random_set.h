#ifndef _FLEXFLOW_LIB_UTILS_BENCHMARK_SRC_UTILS_RANDOM_SET_H
#define _FLEXFLOW_LIB_UTILS_BENCHMARK_SRC_UTILS_RANDOM_SET_H

#include <unordered_set>
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

std::unordered_set<int> random_set(nonnegative_int num_elements);

} // namespace 

#endif
