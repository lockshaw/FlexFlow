/**
 * @brief Utilities for generating random DAGs based on the NASNet-A
 * architecture. NASNet-A is composed of a series of cells, which we randomly
 * generate.
 *
 * For context, see:
 * - Paper: https://arxiv.org/abs/1902.09635
 * - Reference implementation:
 * https://github.com/google-research/nasbench/blob/b94247037ee470418a3e56dcb83814e9be83f3a8/nasbench/api.py
 */

#include "utils/containers/all_of.h"
#include "utils/containers/repeat.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/digraph/algorithms/get_initial_nodes.h"
#include "utils/graph/digraph/algorithms/get_terminal_nodes.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/series_parallel/digraph_generation.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <optional>
#include <vector>

namespace FlexFlow {

const nonnegative_int MIN_NODES = nonnegative_int{6};
const nonnegative_int MAX_NODES = nonnegative_int{8};
const nonnegative_int MIN_EDGES = nonnegative_int{8};
const nonnegative_int MAX_EDGES = nonnegative_int{11};
const nonnegative_int NUM_CELLS = nonnegative_int{9};

struct NasNetBenchConfig {
  std::vector<std::vector<bool>> adjacency_matrix;
};

bool is_valid_config(NasNetBenchConfig const &config);

bool is_valid_cell(DiGraphView const &g);

std::optional<DiGraph>
    maybe_generate_nasnet_bench_cell(NasNetBenchConfig const &config);

DiGraph generate_nasnet_bench_cell();

DiGraph generate_nasnet_bench_network();

} // namespace FlexFlow
