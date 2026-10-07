#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_HEX_STRING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_HEX_STRING_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <random>
#include <string>

namespace FlexFlow {

std::string random_hex_string(std::mt19937 &gen, nonnegative_int size);

} // namespace FlexFlow

#endif
