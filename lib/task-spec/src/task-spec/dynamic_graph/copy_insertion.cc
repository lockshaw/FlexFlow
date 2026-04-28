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
#include "task-spec/dynamic_graph/parallel_tensor_mapping.dtg.h"
#include "utils/containers/values.h"
#include "utils/containers/map_values.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/count.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.h"
#include "utils/containers/get_only.h"
#include "task-spec/dynamic_graph/copy_insertion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/filter_values.h"

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
  result.mapping = ParallelTensorMapping{
    get_tensor_bindings_for_slot_name(mapping, slot.pcg_slot_name),
  };
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

static bool invocation_should_start_mapped(DynamicNodeInvocation const &i)
{
  TrainingOpType op_type = dynamic_node_invocation_get_op_type(i);
  return training_op_type_should_start_mapped(op_type);
}

static std::pair<DynamicValueAttrs, DynamicValueAttrs>
    filter_mapping_to_avoid_degenerate_copies(DynamicValueAttrs const &input,
                                              DynamicValueAttrs const &output) {
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      input_mapping = unordered_set_of(assert_unwrap(input.mapping).raw);

  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      output_mapping = unordered_set_of(assert_unwrap(output.mapping).raw);

  // Exclude the point shared between the input and output mappings, because
  // those will not result in actual copies once shard expansion is performed
  std::unordered_set<
      std::pair<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>>
      remove = set_intersection(input_mapping, output_mapping);

  DynamicValueAttrs filtered_input = input;
  filtered_input.mapping = ParallelTensorMapping{
    bidict_from_pairs(set_difference(input_mapping, remove)),
  };

  DynamicValueAttrs filtered_output = output;
  filtered_output.mapping = ParallelTensorMapping{
    bidict_from_pairs(set_difference(output_mapping, remove)),
  };

  return std::pair{filtered_input, filtered_output};
}

//std::unordered_set<DynamicNodeInvocation> perform_value_mapping_for_invocation(
    //DynamicNodeInvocation const &i)
//{
  //{
    //bool should_start_mapped = invocation_should_start_mapped(i);
    //ASSERT(should_start_mapped == i.node_attrs.mapping.has_value());

    //if (!should_start_mapped) {
      //return {i};
    //}
  //}

  //MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  //auto map_tensor = [&](DynamicTensorSlot const &slot,
                        //DynamicValueAttrs const &value)
    //-> DynamicValueAttrs
  //{
    //return map_dynamic_value_attrs_for_task_group(slot, value, mapping);
  //};

  //std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> mapped_inputs =
      //map_values2(i.inputs, map_tensor);

  //std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> mapped_outputs =
      //map_values2(i.outputs, map_tensor);

  //auto generate_copy_operation_for_input =
    //[&](DynamicTensorSlot const &slot, DynamicValueAttrs const &input)
      //-> std::optional<DynamicNodeInvocation>
    //{
      //DynamicValueAttrs mapped_outgoing_value =
          //unmapped_value_to_mapped_outgoing_value.at(input);

      //DynamicValueAttrs use_value = mapped_inputs.at(slot);

      //if (mapped_outgoing_value == use_value) {
        //return std::nullopt;
      //}

      //auto [filtered_source, filtered_use] =
          //filter_mapping_to_avoid_degenerate_copies(source_value, use_value);

      //return DynamicNodeInvocation{
          //[>inputs=<]{
              //{
                  //DynamicTensorSlot{TensorSlotName::INPUT,
                                    //slot.slot_tensor_role},
                  //filtered_source,
              //},
          //},
          //[>node_attrs=<]
          //DynamicNodeAttrs{
              //[>task_type=<]transform(
                  //slot.slot_tensor_role,
                  //dynamic_task_type_from_tensor_role_for_copy),
              //[>device_coord=<]std::nullopt,
              //[>mapping=<]std::nullopt,
              //[>op_attrs<] TrainingOperationAttrs{CopyAttrs{}},
              //[>layer_guid=<]dynamic_layer_guid_t{dynamic_copy_layer_guid_t{}},
              //[>per_device_op_state=<]std::nullopt,
          //},
          //[>outputs=<]
          //{
              //{
                  //DynamicTensorSlot{TensorSlotName::OUTPUT,
                                    //slot.slot_tensor_role},
                  //filtered_use,
              //},
          //},
      //};
    //};

  //std::unordered_set<DynamicNodeInvocation> copy_invocations =
    //filtrans(
      //unordered_set_of(i.inputs),
      //[&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p)
        //-> std::optional<DynamicNodeInvocation>
      //{
        //return generate_copy_operation_for_input(p.first, p.second);
      //});

  //return set_union(
    //copy_invocations,
    //std::unordered_set{
      //DynamicNodeInvocation{
        //[>inputs=<]mapped_inputs,
        //[>node_attrs=<]i.node_attrs,
        //[>outputs=<]mapped_outputs,
      //},
    //});
//}


std::unordered_map<InternalDynamicSlotSite, ParallelTensorMapping>
  solve_for_unresolved_mappings(
    DynamicOpenDataflowGraph const &g,
    std::unordered_map<InternalDynamicSlotSite, std::optional<ParallelTensorMapping>> const &partial_solution)
{
  std::unordered_map<InternalDynamicSlotSite, std::optional<ParallelTensorMapping>> curr_solution = partial_solution;

  auto get_unresolved = [&]() -> std::unordered_set<InternalDynamicSlotSite> {
    return keys(filter_values(curr_solution,
                              [](std::optional<ParallelTensorMapping> const &m) -> bool {
                                return !m.has_value();
                              }));
  };

  auto count_unresolved = [&]() -> nonnegative_int {
    return num_elements(get_unresolved());
  };

  auto try_to_resolve = [&](InternalDynamicSlotSite const &s)
    -> std::optional<ParallelTensorMapping>
  {
    ASSERT(curr_solution.at(s) == std::nullopt);

    DynamicNodeInvocation i = s.invocation;
    DynamicValueAttrs v = dynamic_value_attrs_for_slot_site(DynamicSlotSite{s});

    if (s.direction == TensorDirection::OUTPUT) {
      std::unordered_set<InternalDynamicSlotSite> sinks = dynamic_graph_find_sinks_of_value(g, v);

      if (sinks.size() > 0) {
        std::unordered_set<std::optional<ParallelTensorMapping>> sink_solutions =
          transform(sinks, [&](InternalDynamicSlotSite const &sink) { return curr_solution.at(sink); });
        sink_solutions.erase(std::nullopt);

        if (sink_solutions.size() == 1) {
          return get_only(sink_solutions);
        }
      }
    }

    if (s.direction == TensorDirection::INCOMING) {
      DynamicSlotSite source = dynamic_graph_find_source_of_value(g, v);

      if (source.is_internal()) {
        std::optional<ParallelTensorMapping> source_mapping = curr_solution.at(source.require_internal());
        if (source_mapping.has_value()) {
          return source_mapping.value();
        }
      }
    }

    // currently failing because different inputs to the update task don't influence each other, so the 
    // sgd_v value remains unmapped

  };

  auto try_to_resolve_all_unresolved = [&]() -> bool {
    bool made_progress = false;
    for (auto const &[slot_site, mapping] : curr_solution) {
      if (mapping == std::nullopt) {
        std::optional<ParallelTensorMapping> resolution = try_to_resolve(slot_site);

        if (resolution.has_value()) {
          curr_solution.at(slot_site) = resolution;
          made_progress = true;
        }
      }
    }

    return made_progress;
  };

  while (count_unresolved() > 0) {
    bool made_progress = try_to_resolve_all_unresolved();

    ASSERT(made_progress, get_unresolved());
  }

  return
    map_values(
      curr_solution,
      [](std::optional<ParallelTensorMapping> const &m) -> ParallelTensorMapping {
        return m.value();
      });
}

std::unordered_map<InternalDynamicSlotSite, ParallelTensorMapping>
  resolve_tensor_mappings_from_node_mappings(DynamicOpenDataflowGraph const &g) {

  auto get_mappings_for_invocation = [&](DynamicNodeInvocation const &i)
    -> std::unordered_map<InternalDynamicSlotSite, std::optional<ParallelTensorMapping>>
  {
    auto get_tensor_mapping_for_slot_name = [&](TensorSlotName slot_name)
      -> std::optional<ParallelTensorMapping>
    {
      if (i.node_attrs.mapping.has_value()) {
        return ParallelTensorMapping{
          get_tensor_bindings_for_slot_name(assert_unwrap(i.node_attrs.mapping), slot_name),
        };
      } else {
        return std::nullopt;
      }
    };

    return generate_map(
      get_dynamic_slot_sites_for_invocation(i),
      [&](InternalDynamicSlotSite const &s) {
        return get_tensor_mapping_for_slot_name(s.slot_name.pcg_slot_name);
      });
  };

  std::unordered_map<
    InternalDynamicSlotSite,
    std::optional<ParallelTensorMapping>
  > initial_partial_solution =
    merge_disjoint_maps(
      transform(
        get_dynamic_invocation_set(g),
        get_mappings_for_invocation));

  std::unordered_map<
    InternalDynamicSlotSite,
    ParallelTensorMapping
  > result = solve_for_unresolved_mappings(g, initial_partial_solution);

  return result;
}

std::unordered_map<DynamicTensorSlot, ParallelTensorMapping>
  get_mappings_for_invocation(
    DynamicNodeInvocation const &i,
    std::unordered_map<InternalDynamicSlotSite, ParallelTensorMapping> const &mappings)
{
  return filtermap_keys(
    mappings,
    [&](InternalDynamicSlotSite const &s) -> std::optional<DynamicTensorSlot> {
      if (s.invocation == i) {
        return s.slot_name;
      } else {
        return std::nullopt;
      }
    });
}

DynamicNodeInvocation apply_mappings_for_invocation(
  DynamicNodeInvocation const &i,
  std::unordered_map<InternalDynamicSlotSite, ParallelTensorMapping> const &all_mappings)
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
      zip_values_strict_with(
        i.inputs,
        i_input_mappings,
        apply_mapping),
    /*node_attrs=*/
      i.node_attrs,
    /*outputs=*/
      zip_values_strict_with(
        i.outputs,
        i_output_mappings,
        apply_mapping),
  };
}

std::unordered_set<DynamicNodeInvocation>
  copies_for_value(DynamicOpenDataflowGraph const &g,
                   DynamicValueAttrs const &v,
                   std::unordered_map<InternalDynamicSlotSite, ParallelTensorMapping> const &mappings)
{
  InternalDynamicSlotSite src = ({
    DynamicSlotSite found = dynamic_graph_find_source_of_value(g, v);

    if (found.is_external()) {
      return {};
    }

    found.require_internal();
  });

  std::unordered_set<InternalDynamicSlotSite> sinks = dynamic_graph_find_sinks_of_value(g, DynamicValueAttrs{v});

  ParallelTensorMapping src_mapping = mappings.at(src);

  std::unordered_map<InternalDynamicSlotSite, ParallelTensorMapping> mappings_for_sinks =
    generate_map(sinks,
                 [&](InternalDynamicSlotSite const &s) -> ParallelTensorMapping {
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
                  src.slot_name.slot_tensor_role,
                },
                dynamic_value_attrs_with_mapping(v, src_mapping),
            },
        },
        /*node_attrs=*/
        DynamicNodeAttrs{
            /*task_type=*/transform(
                src.slot_name.slot_tensor_role,
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
                  src.slot_name.slot_tensor_role,
                },
                dynamic_value_attrs_with_mapping(v, sink_mapping),
            },
        },
    };
  };

  return transform(required_copies, make_copy_to);
}

DynamicOpenDataflowGraph
    perform_copy_insertion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_copy_inserted(g));

  std::unordered_map<InternalDynamicSlotSite, ParallelTensorMapping>
    fully_resolved_tensor_mappings =
      resolve_tensor_mappings_from_node_mappings(g);

  std::unordered_set<DynamicNodeInvocation>
    all_copies = flatmap(
      unordered_set_of(get_dynamic_values(g)),
      [&](DynamicValueAttrs const &v) -> std::unordered_set<DynamicNodeInvocation> {
        return copies_for_value(g, v, fully_resolved_tensor_mappings);
      });

  std::unordered_set<DynamicNodeInvocation>
    mapped_invocations =
      transform(
        get_dynamic_invocation_set(g),
        [&](DynamicNodeInvocation const &i) -> DynamicNodeInvocation {
          return apply_mappings_for_invocation(i, fully_resolved_tensor_mappings);
        });

  DynamicOpenDataflowGraph result = DynamicOpenDataflowGraph{
    set_union(all_copies, mapped_invocations),
  };

  ASSERT(graph_is_fully_copy_inserted(result));

  return result;
}

} // namespace FlexFlow
