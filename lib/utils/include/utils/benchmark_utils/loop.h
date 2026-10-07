#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_LOOP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_LOOP_H

#include <cstdint>

namespace FlexFlow {

#define LOOP(n, skip) for (volatile uint64_t i = 0; i < (skip ? 0 : n); i++)

} // namespace FlexFlow

#endif
