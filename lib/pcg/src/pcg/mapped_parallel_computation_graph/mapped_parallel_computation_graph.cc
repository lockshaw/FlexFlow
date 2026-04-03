#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/find_isomorphism_between_kwarg_dataflow_graphs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"

namespace FlexFlow {

std::string format_as(MappedParallelComputationGraph const &mapped_pcg) {
  return fmt::format(
      "<MappedParallelComputationGraph\npcg={}\nmapped_tasks={}>",
      as_dot(mapped_pcg.pcg),
      mapped_pcg.mapped_tasks);
}

std::ostream &operator<<(std::ostream &s,
                         MappedParallelComputationGraph const &mapped_pcg) {
  return (s << fmt::to_string(mapped_pcg));
}

bool mapped_pcgs_are_isomorphic(MappedParallelComputationGraph const &src,
                                MappedParallelComputationGraph const &dst) {
  bidict<Node, Node> raw_isomorphism = ({
    std::optional<bidict<Node, Node>> maybe_isomorphism = find_isomorphism_between_kwarg_dataflow_graphs(
       without_layer_names(src.pcg).raw_graph,
       without_layer_names(dst.pcg).raw_graph);
    if (!maybe_isomorphism.has_value()) {
      return false;
    }

    maybe_isomorphism.value();
  });

  auto type_node = [](Node const &n) -> parallel_layer_guid_t {
    return parallel_layer_guid_t{n};
  };

  bidict<parallel_layer_guid_t, parallel_layer_guid_t>
    typed_isomorphism = transform_keys(
      transform_values(raw_isomorphism, type_node),
      type_node);

  auto src_to_dst_layer = [&](parallel_layer_guid_t src_layer) -> parallel_layer_guid_t {
    return typed_isomorphism.at_l(src_layer);
  };

  return map_keys(src.mapped_tasks, src_to_dst_layer) == dst.mapped_tasks;
}

} // namespace FlexFlow
