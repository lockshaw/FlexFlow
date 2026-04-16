#include "task-spec/dynamic_graph/copy_insertion.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_task_type.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/algorithms/bidict_from_pairs.h"
#include "utils/bidict/algorithms/unordered_set_of.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/set_intersection.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/set_difference.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/filtrans.h"
#include "utils/overload.h"

namespace FlexFlow {

bool node_is_copy(DynamicNodeAttrs const &n) {
  return n.op_attrs.has_value() && n.op_attrs.value().is_copy();
}

bool value_is_mapped(DynamicValueAttrs const &n) {
  return n.mapping.has_value();
}

bool no_part_of_graph_is_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return false; };

  return no_part_of_dynamic_graph_satisfies(
      g, node_is_copy, value_is_mapped, slot_is_mapped);
}

bool graph_is_fully_copy_inserted(DynamicOpenDataflowGraph const &g) {
  auto node_is_any = [](DynamicNodeAttrs const &) -> bool { return true; };
  auto slot_is_mapped = [](DynamicTensorSlot const &) -> bool { return true; };

  return full_dynamic_graph_satisfies(
      g, node_is_any, value_is_mapped, slot_is_mapped);
}

static DynamicValueAttrs map_dynamic_value_attrs_for_task_group(
    DynamicTensorSlot const &slot,
    DynamicValueAttrs const &value,
    MappedOperatorTaskGroup const &mapping) {
  DynamicValueAttrs result = value;
  result.mapping = get_tensor_bindings_for_slot_name(mapping, slot.slot_name);
  return result;
}

static bool training_op_type_should_start_mapped(TrainingOpType const &training_op_type) {
  return training_op_type.visit<bool>(overload {
    [](OperatorType const &t) -> bool {
      return should_be_mapped(t);
    },
    [](TrainingOnlyOpType const &t) -> bool {
      ASSERT(t == TrainingOnlyOpType::LOSS);

      return true;
    },
  });
}

static bool invocation_should_start_mapped(TrainingOpType const &training_op_type) {
  TrainingOpType op_type = dynamic_node_invocation_get_op_type(i);
  return training_op_type_should_start_mapped(op_type);
}

static std::pair<DynamicValueAttrs, DynamicValueAttrs>
    filter_mapping_to_avoid_degenerate_copies(DynamicValueAttrs const &input,
                                              DynamicValueAttrs const &output) {
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      input_mapping = unordered_set_of(assert_unwrap(input.mapping));

  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      output_mapping = unordered_set_of(assert_unwrap(output.mapping));

  // Exclude the point shared between the input and output mappings, because
  // those will not result in actual copies once shard expansion is performed
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      remove = set_intersection(input_mapping, output_mapping);

  DynamicValueAttrs filtered_input = input;
  filtered_input.mapping =
      bidict_from_pairs(set_difference(input_mapping, remove));

  DynamicValueAttrs filtered_output = output;
  filtered_output.mapping =
      bidict_from_pairs(set_difference(output_mapping, remove));

  return std::pair{filtered_input, filtered_output};
}

std::unordered_set<DynamicNodeInvocation> perform_value_mapping_for_invocation(
    DynamicNodeInvocation const &i,
{
  {
    bool should_start_mapped = invocation_should_start_mapped(i);
    ASSERT(should_start_mapped == i.node_attrs.mapping.has_value());

    if (!should_start_mapped) {
      return {i};
    }
  }

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  auto map_tensor = [&](DynamicTensorSlot const &slot,
                        DynamicValueAttrs const &value) 
    -> DynamicValueAttrs
  {
    return map_dynamic_value_attrs_for_task_group(slot, value, mapping);
  };

  std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> mapped_inputs =
      map_values2(i.inputs, map_tensor);

  std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> mapped_outputs =
      map_values2(i.outputs, map_tensor);

  auto generate_copy_operation_for_input = 
    [&](DynamicTensorSlot const &slot, DynamicValueAttrs const &input) 
      -> std::optional<DynamicNodeInvocation>
    {
      if (!contains_key(unmapped_value_to_mapped_outgoing_value, input)) {
        return std::nullopt;
      }

      DynamicValueAttrs mapped_outgoing_value =
          unmapped_value_to_mapped_outgoing_value.at(input);

      DynamicValueAttrs use_value = mapped_inputs.at(slot);

      if (mapped_outgoing_value == use_value) {
        return std::nullopt;
      }

      auto [filtered_source, filtered_use] =
          filter_mapping_to_avoid_degenerate_copies(source_value, use_value);

      return DynamicNodeInvocation{
          /*inputs=*/{
              {
                  DynamicTensorSlot{TensorSlotName::INPUT,
                                    slot.slot_tensor_role},
                  filtered_source,
              },
          },
          /*node_attrs=*/
          DynamicNodeAttrs{
              /*task_type=*/transform(
                  slot.slot_tensor_role,
                  dynamic_task_type_from_tensor_role_for_copy),
              /*device_coord=*/std::nullopt,
              /*mapping=*/std::nullopt,
              /*op_attrs*/ TrainingOperationAttrs{CopyAttrs{}},
              /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
              /*per_device_op_state=*/std::nullopt,
          },
          /*outputs=*/
          {
              {
                  DynamicTensorSlot{TensorSlotName::OUTPUT,
                                    slot.slot_tensor_role},
                  filtered_use,
              },
          },
      };
    };

  std::unordered_set<DynamicNodeInvocation> copy_invocations = 
    filtrans(
      unordered_set_of(i.inputs),
      [&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p) 
        -> std::optional<DynamicNodeInvocation>
      {
        return generate_copy_operation_for_input(p.first, p.second);
      });

  return set_union(
    copy_invocations,
    std::unordered_set{
      DynamicNodeInvocation{
        /*inputs=*/mapped_inputs,
        /*node_attrs=*/i.node_attrs,
        /*outputs=*/mapped_outputs,
      },
    });
}


std::unordered_map<DynamicNodeSlot, ParallelTensorMapping>
  resolve_tensor_mappings_from_node_mappings(DynamicOpenDataflowGraph const &g) {

  std::unordered_map<
    DynamicNodeSlot, 
    std::optional<ParallelTensorMapping>
  > result;

  for (DynamicNodeInvocation const &i : get_dynamic_invocation_set(g)) {
    if (i.node_attrs.mapping.has_value()) {
      for (DynamicNodeSlot const &s : get_dynamic_node_slots_for_invocation(i)) {
        result.insert({
          s, 
          get_tensor_bindings_for_slot_name(mapping, slot.slot_name.slot_name),
        });
      }
    } else {
      for (DynamicNodeSlot const &s : get_dynamic_node_slots_for_invocation(i)) {
        DynamicValueAttrs val = dynamic_value_attrs_for_node_slot(s);
        result.insert({s, std::nullopt});
      }
    }
  }

  auto count_unresolved = [&]() -> nonnegative_int {
    return count(values(result), 
                 [](std::optional<ParallelTensorMapping> const &m) {
                   return m == std::nullopt; 
                 });
  };

  auto try_to_resolve = [&](DynamicNodeSlot const &s) 
    std::optional<bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
  {
    ASSERT(result.at(s) == std::nullopt);

    DynamicNodeInvocation i = s.invocation;
    DynamicValueAttrs v = dynamic_value_attrs_for_node_slot(s);

    if (s.direction == TensorDirection::OUTGOING) {
      std::unordered_set<DynamicNodeSlot> sinks = dynamic_graph_find_sinks_of_value(g, v);

      if (sinks.size() == 0) {
        return std::nullopt;
      }

      DynamicNodeSlot sink = get_only(sinks);

      return result.at(sink);
    } else {
      ASSERT(s.direction == TensorDirection::INCOMING);

      DynamicNodeSlot source = dynamic_graph_find_source_of_value(g, v);

      return result.at(source);
    }
  };

  auto try_to_resolve_all_unresolved = [&]() -> bool {
    bool made_progress = true;
    for (auto const &[node_slot, mapping] : result) {
      if (mapping == std::nullopt) {
        std::optional<ParallelTensorMapping> resolution = try_to_resolve(node_slot);

        if (resolution.has_value()) {
          result.at(node_slot) = resolution;
        }
      }
    }

    return made_progress;
  };

  while (count_unresolved() > 0) {
    bool made_progress = try_to_resolve_all_unresolved();

    ASSERT(made_progress);
  }

  return 
    map_values(
      result, 
      [](std::optional<ParallelTensorMapping> const &m) -> ParallelTensorMapping {
        return m.value();
      });
}

std::unordered_map<DynamicTensorSlot, ParallelTensorMapping> 
  get_mappings_for_invocation(
    DynamicNodeInvocation const &i,
    std::unordered_map<DynamicNodeSlot, ParallelTensorMapping> const &mappings)
{
  return filtermap_keys(
    mappings,
    [&](DynamicNodeSlot const &s) -> std::optional<DynamicTensorSlot> {
      if (s.invocation == i) {
        return s.slot_name;
      } else {
        return std::nullopt;
      }
    });
}


DynamicNodeInvocation apply_mappings_for_invocation(
  DynamicNodeInvocation const &i,
  std::unordered_map<DynamicNodeSlot, ParallelTensorMapping> const &all_mappings) 
{
  std::unordered_map<DynamicTensorSlot, ParallelTensorMapping> i_mappings = 
    get_mappings_for_invocation(i, all_mappings);

  std::unordered_map<DynamicTensorSlot, ParallelTensorMapping> i_input_mappings = 
    restrict_keys(i_mappings, keys(i.inputs));

  std::unordered_map<DynamicTensorSlot, ParallelTensorMapping> i_output_mappings = 
    restrict_keys(i_mappings, keys(i.outputs));

  auto apply_mapping = 
    [&](DynamicValueAttrs const &v, ParallelTensorMapping const &mapping) 
      -> DynamicValueAttrs
    {
      return dynamic_value_attrs_with_mapping(v, mapping);
    };

  return DynamicNodeInvocation{
    /*inputs=*/
      zip_values_with(
        i.inputs,
        i_input_mappings, 
        apply_mapping),
    /*node_attrs=*/
      i.node_attrs,
    /*outputs=*/
      zip_values_with(
        i.outputs,
        output_mappings, 
        apply_mapping),
  };
}

std::unordered_set<DynamicNodeInvocation>
  copies_for_value(DynamicOpenDataflowGraph const &g, 
                   DynamicValueAttrs const &v,
                   std::unordered_set<DynamicNodeSlot, ParallelTensorMapping> const &mappings)  
{
  DynamicNodeSlot src = dynamic_graph_find_source_of_value(g, v);
  std::unordered_set<DynamicNodeSlot> sinks = dynamic_graph_find_sinks_of_value(g, v);

  ParallelTensorMapping src_mapping = mappings.at(src);
  
  std::unordered_map<DynamicNodeSlot, ParallelTensorMapping> mappings_for_sinks =
    generate_map(sinks, 
                 [&](DynamicNodeSlot const &s) -> ParallelTensorMapping {
                   return mappings.at(s);
                 });

  std::unordered_set<ParallelTensorMapping> sink_mapping_set = unordered_set_of(values(mappings_for_sinks));

  std::unordered_set<ParallelTensorMapping> required_copies = set_difference(sink_mapping_set, std::unordered_set{src_mapping});

  auto make_copy_to = [&](ParallelTensorMapping const &sink_mapping) 
    -> DynamicNodeInvocation
  {
    return DynamicNodeInvocation{
        /*inputs=*/{
            {
                DynamicTensorSlot{
                  TensorSlotName::INPUT,
                  slot.slot_tensor_role,
                },
                dynamic_value_attrs_with_mapping(v, src_mapping);
            },
        },
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/transform(
                slot.slot_tensor_role,
                dynamic_task_type_from_tensor_role_for_copy),
            /*device_coord=*/std::nullopt,
            /*mapping=*/std::nullopt,
            /*op_attrs*/ TrainingOperationAttrs{CopyAttrs{}},
            /*layer_guid=*/dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
            /*per_device_op_state=*/std::nullopt,
        },
        /*outputs=*/
        {
            {
                DynamicTensorSlot{
                  TensorSlotName::OUTPUT,
                  slot.slot_tensor_role,
                },
                dynamic_value_attrs_with_mapping(v, sink_mapping);
            },
        },
    };
  };

}

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_copy_inserted(g));

  auto map_values_for_invocation_and_direction = 
    [](DynamicNodeInvocation const &i, TensorDirection direction) 
      -> std::unordered_map<DynamicValueAttrs, std::optional<DynamicValueAttrs>> 
    {
      bool should_start_mapped = invocation_should_start_mapped(i);
      ASSERT(should_start_mapped == i.node_attrs.mapping.has_value());

      auto map_value = 
        [&](DynamicTensorSlot const &slot, DynamicValueAttrs const &v) 
          -> std::optional<DynamicValueAttrs> 
        {
          if (i.node_attrs.mapping.has_value()) {
            return map_dynamic_value_attrs_for_task_group(
              slot, v, assert_unwrap(i.node_attrs.mapping));
          } else {
            return std::nullopt;
          }
        };

      return transform(
        get_slot_map_for_direction(i.outputs, direction), 
        [&](DynamicTensorSlot const &slot, DynamicValueAttrs const &unmapped) {
          return std::pair{
            unmapped,
            map_value(slot, unmapped),
          };
        });
    };

  auto map_values_for_direction = [&](TensorDirection direction) 
    -> std::unordered_map<DynamicValueAttrs, std::optional<DynamicValueAttrs>>
  {
    return merge_disjoint_maps(
      transform(g.invocations, 
                [&](DynamicNodeInvocation const &i) 
                  -> std::unordered_map<DynamicValueAttrs, std::optional<DynamicValueAttrs>>
                {
                  return map_values_output_by_invocation(i, direction);
                }));
  };


  std::unordered_map<DynamicValueAttrs, DynamicValueAttrs>
      unmapped_value_to_mapped_outgoing_value = 
        map_values_for_direction(TensorDirection::OUTPUT);

  std::unordered_map<DynamicValueAttrs, DynamicValueAttrs>
      unmapped_value_to_mapped_incoming_value = 
        map_values_for_direction(TensorDirection::INCOMING);



  // Use regular flatmap here to remove duplicates (we don't want to copy the
  // same tensor to the same place multiple times)
  DynamicOpenDataflowGraph result =
      dynamic_open_dataflow_graph_from_invocation_set(
          flatmap(g.invocations, [&](DynamicNodeInvocation const &i) {
            return perform_copy_insertion_for_invocation(
                i, unmapped_value_to_mapped_source_value);
          }));

  ASSERT(graph_is_fully_copy_inserted(result));

  return result;
}

} // namespace FlexFlow
