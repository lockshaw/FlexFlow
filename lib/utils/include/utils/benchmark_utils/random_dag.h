#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_DAG_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BENCHMARK_UTILS_RANDOM_DAG_H

#include "utils/graph/digraph/digraph.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <random>

namespace FlexFlow {

void random_dag(std::mt19937 &gen,
                DiGraph &g,
                nonnegative_int num_nodes,
                nonnegative_int num_edges);

} // namespace FlexFlow

#endif
