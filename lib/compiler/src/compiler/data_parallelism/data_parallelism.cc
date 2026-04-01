#include "compiler/data_parallelism/data_parallelism.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution.dtg.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/zip_values_strict.h"

namespace FlexFlow {

SearchResult apply_data_parallelism(ComputationGraph const &cg,
                                    int_ge_two degree) {
  ParallelComputationGraph pcg = empty_parallel_computation_graph();
  MachineMapping machine_mapping = MachineMapping{{}};

  MachineView data_parallel_mv = MachineView{
    /*start=*/MachineSpaceCoordinate{
      /*node_idx=*/0_n,
      /*device_idx=*/0_n,
    },
    /*dimensions=*/{

    },
  };

  bidict<tensor_guid_t, parallel_tensor_guid_t>
    cg_tensor_to_pcg_tensor = {};

  auto add_layer_to_pcg = [&](layer_guid_t const &layer) -> void {
    ParallelLayerAttrs parallel_layer_attrs =
      parallel_layer_attrs_from_layer_attrs(get_layer_attrs(cg, layer));

    ParallelLayerAddedResult added = add_parallel_layer(
      /*pcg=*/pcg,
      /*layer_attrs=*/parallel_layer_attrs,
      /*inputs=*/map_values(
        get_incoming_inputs(cg, layer),
        [&](tensor_guid_t t) -> parallel_tensor_guid_t {
          return cg_tensor_to_pcg_tensor.at_l(t);
        }),
      /*weights=*/map_values(
        get_incoming_weights(cg, layer),
        [&](tensor_guid_t t) -> parallel_tensor_guid_t {
          return cg_tensor_to_pcg_tensor.at_l(t);
        }),
      /*outputs=*/map_values(
        get_outgoing_tensors(cg, layer),
          [&](tensor_guid_t t) -> CreateGrad {
            return get_tensor_attrs(cg, t).create_grad;
          }));

    for (
      std::pair<tensor_guid_t, parallel_tensor_guid_t> const &corresponding_outputs
      : values(zip_values_strict(
          get_outgoing_tensors(cg, layer),
          added.outputs))
    ) {
      cg_tensor_to_pcg_tensor.equate_strict(corresponding_outputs);
    }
  };

  for (layer_guid_t const &layer : topological_ordering(cg)) {
    add_layer_to_pcg(layer);
  }

  return SearchResult{
    /*pcg=*/pcg,
    /*machine_mapping=*/machine_mapping,
  };
}

} // namespace FlexFlow
