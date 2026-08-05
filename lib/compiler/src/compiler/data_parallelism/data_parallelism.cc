#include "compiler/data_parallelism/data_parallelism.h"
#include "op-attrs/parallel_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/shape_inference.h"
#include "pcg/computation_graph.h"
#include "pcg/parallel_computation_graph/generate_weight_transform.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution.dtg.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/containers/zip_values_strict_with.h"

namespace FlexFlow {

SearchResult apply_data_parallelism(ComputationGraph const &cg,
                                    int_ge_two degree) {
  ParallelComputationGraph pcg = empty_parallel_computation_graph();

  MachineView input_mv = MachineView{
      /*start=*/MachineSpaceCoordinate{
          /*node_idx=*/0_n,
          /*device_idx=*/0_n,
      },
      /*start_invariant=*/
      StartInvariantMachineView{
          MachineView1dProjection{
              /*strides=*/{},
          },
      },
  };

  MachineView data_parallel_mv = MachineView{
      /*start=*/MachineSpaceCoordinate{
          /*node_idx=*/0_n,
          /*device_idx=*/0_n,
      },
      /*start_invariant=*/
      StartInvariantMachineView{
          MachineView1dProjection{
              /*strides=*/{stride_t{1_p}},
          },
      },
  };

  bidict<tensor_guid_t, parallel_tensor_guid_t> cg_tensor_to_pcg_tensor = {};

  auto parallelize_tensor =
      [&](parallel_tensor_guid_t t,
          ParallelTensorDimDegrees degrees) -> parallel_tensor_guid_t {
    ASSERT(get_total_parallel_degree(get_parallel_tensor_shape(pcg, t)) == 1);

    for (ParallelOpAttrs const &parallel_op :
         generate_weight_transform(degrees)) {
      t = require_only_key(
          pcg_add_parallel_op_layer(pcg, parallel_op, t).outputs,
          TensorSlotName::OUTPUT);
    }

    return t;
  };

  auto add_layer_to_pcg = [&](layer_guid_t const &layer) -> void {
    ParallelLayerAttrs parallel_layer_attrs =
        parallel_layer_attrs_from_layer_attrs(get_layer_attrs(cg, layer));

    std::map<TensorSlotName, parallel_tensor_guid_t> inputs =
        map_values(get_incoming_inputs(cg, layer),
                   [&](tensor_guid_t t) -> parallel_tensor_guid_t {
                     return cg_tensor_to_pcg_tensor.at_l(t);
                   });

    std::map<TensorSlotName, ParallelTensorDimDegrees> input_degrees =
        map_values(
            inputs, [&](parallel_tensor_guid_t t) -> ParallelTensorDimDegrees {
              return get_parallel_degrees(get_parallel_tensor_shape(pcg, t));
            });

    std::map<TensorSlotName, ParallelTensorDimDegrees>
        weight_degrees =
            infer_weight_degrees(parallel_layer_attrs.op_attrs, input_degrees);

    std::map<TensorSlotName, parallel_tensor_guid_t> weights =
        map_values(get_incoming_weights(cg, layer),
                   [&](tensor_guid_t t) -> parallel_tensor_guid_t {
                     return cg_tensor_to_pcg_tensor.at_l(t);
                   });

    std::map<TensorSlotName, parallel_tensor_guid_t>
        parallelized_weights =
            zip_values_strict_with(weights, weight_degrees, parallelize_tensor);

    ParallelLayerAddedResult added = add_parallel_layer(
        /*pcg=*/pcg,
        /*layer_attrs=*/parallel_layer_attrs,
        /*inputs=*/inputs,
        /*weights=*/parallelized_weights,
        /*outputs=*/
        map_values(get_outgoing_tensors(cg, layer),
                   [&](tensor_guid_t t) -> CreateGrad {
                     return get_tensor_attrs(cg, t).create_grad;
                   }));

    if (get_op_type(parallel_layer_attrs) == OperatorType::INPUT) {
      added = add_parallel_layer(
          /*pcg=*/pcg,
          /*layer_attrs=*/
          ParallelLayerAttrs{
              /*op_attrs=*/PCGOperatorAttrs{
                  RepartitionAttrs{
                      ff_dim_t{0_n},
                      degree,
                  },
              },
              /*name=*/std::nullopt,
          },
          /*inputs=*/
          {
              {
                  TensorSlotName::INPUT,
                  require_only_key(added.outputs, TensorSlotName::OUTPUT),
              },
          },
          /*weights=*/{},
          /*outputs=*/
          map_values(get_outgoing_tensors(cg, layer),
                     [&](tensor_guid_t t) -> CreateGrad {
                       return get_tensor_attrs(cg, t).create_grad;
                     }));
    }

    for (std::pair<tensor_guid_t, parallel_tensor_guid_t> const
             &corresponding_outputs : values(zip_values_strict(
                 get_outgoing_tensors(cg, layer), added.outputs))) {
      cg_tensor_to_pcg_tensor.equate_strict(corresponding_outputs);
    }
  };

  for (layer_guid_t const &layer : topological_ordering(cg)) {
    add_layer_to_pcg(layer);
  }

  MachineMapping machine_mapping = MachineMapping{
      generate_map(pcg_get_parallel_layers(pcg),
                   [&](parallel_layer_guid_t const &l) -> MachineView {
                     OperatorType op_type =
                         pcg_op_attrs_get_op_type(pcg_get_op_attrs(pcg, l));
                     if (op_type == OperatorType::INPUT ||
                         op_type == OperatorType::WEIGHT) {
                       return input_mv;
                     } else {
                       return data_parallel_mv;
                     }
                   }),
  };

  return SearchResult{
      /*pcg=*/pcg,
      /*machine_mapping=*/machine_mapping,
  };
}

} // namespace FlexFlow
