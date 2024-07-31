#ifndef _FLEXFLOW_COMPILER_MACHINE_MAPPING_H
#define _FLEXFLOW_COMPILER_MACHINE_MAPPING_H

#include "compiler/machine_mapping.dtg.h"
#include "cost_estimate.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

MachineMapping combine(MachineMapping const &, MachineMapping const &);

bool nodes_are_disjoint(MachineMapping const &m1, MachineMapping const &m2);

struct OptimalCostState {
  SerialParallelDecomposition subgraph;
  MachineSpecification resource;
  std::unordered_map<Node, MachineView> given_machine_views;
  req<std::unordered_map<OpenDataflowEdge, MachineView>> frontier_machine_views;
};
FF_VISITABLE_STRUCT(OptimalCostState,
                    subgraph,
                    resource,
                    given_machine_views,
                    frontier_machine_views);

struct OptimalCostResult {
  static OptimalCostResult sequential_combine(OptimalCostResult const &s1,
                                              OptimalCostResult const &s2);
  static OptimalCostResult parallel_combine(OptimalCostResult const &s1,
                                            OptimalCostResult const &s2);
  static OptimalCostResult infinity();

  float runtime;
  req<MachineMapping> machine_mapping;
};
FF_VISITABLE_STRUCT(OptimalCostResult, runtime, machine_mapping);

struct OptimalCostRuntimeCmp {
  bool operator()(OptimalCostResult const &, OptimalCostResult const &);
};

class OptimalCostCache {
public:
  OptimalCostCache() = default;

  std::optional<OptimalCostResult> load(OptimalCostState const &) const;
  void save(OptimalCostState const &, OptimalCostResult const &);

private:
  std::unordered_map<OptimalCostState, OptimalCostResult> cache;
};

OptimalCostResult optimal_cost(
    ParallelComputationGraph const &g,
    std::function<std::unordered_set<MachineView>(
        ParallelLayerAttrs const &, MachineSpecification const &)> const
        &allowed_machine_views,
    CostEstimator const &cost_estimator,
    MachineSpecification const &resources,
    OptimalCostCache &cached_subgraph_costs);

} // namespace FlexFlow

namespace std {

template <>
struct hash<std::unordered_map<FlexFlow::Node, FlexFlow::MachineMapping>> {
  size_t operator()(
      std::unordered_map<FlexFlow::Node, FlexFlow::MachineMapping> const &g)
      const;
};

}; // namespace std

#endif
