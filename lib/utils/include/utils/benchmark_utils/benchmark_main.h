#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_H

#include <functional>
#include <map>
#include <string>

namespace FlexFlow {

void benchmark_main(
    int argc,
    char **argv,
    std::map<std::string, std::function<void(bool)>> const &benchmarks);

} // namespace FlexFlow

#endif
