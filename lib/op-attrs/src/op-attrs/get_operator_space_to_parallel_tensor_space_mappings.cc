#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/get_tensor_roles.h"
#include "op-attrs/incoming_tensor_role.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/gather.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/noop.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/reduce.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/reshape.h"
#include "op-attrs/ops/reverse.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/split.h"
#include "op-attrs/ops/topk.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/upsample.h"
#include "op-attrs/ops/weight.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_three_keys.h"
#include "utils/containers/require_two_keys.h"
#include "utils/containers/restrict_keys_strict.h"
#include "utils/containers/slice.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename T>
static std::vector<T>
    require_only_slots_sequence(std::map<TensorSlotName, T> const &v,
                                std::vector<TensorSlotName> const &slots) {
  nonnegative_int v_num_slots = num_elements(v);
  ASSERT(v_num_slots <= slots.size());

  std::vector<TensorSlotName> expected_slots =
      slice(slots, 0, v_num_slots.unwrap_nonnegative());

  ASSERT(set_of(expected_slots) == keys(v));

  return transform(expected_slots, [&](TensorSlotName const &slot_name) {
    return v.at(slot_name);
  });
};

static std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    lift_biunique_mappings(
        std::map<TensorSlotName,
                 OperatorSpaceToParallelTensorSpaceBiuniqueMapping> const
            &biunique) {
  return map_values(biunique, operator_ptensor_space_mapping_from_biunique);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings(
        PCGOperatorAttrs const &op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {

  return binary_merge_disjoint_maps(
      get_operator_to_input_mappings(op_attrs, inputs_degrees),
      get_operator_to_weight_mappings(op_attrs, inputs_degrees));
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings_for_role(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        IncomingTensorRole incoming_tensor_role) {
  return get_operator_to_ptensor_mappings_for_role(
      attrs,
      inputs_degrees,
      tensor_role_from_incoming_tensor_role(incoming_tensor_role));
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_input_mappings(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return get_operator_to_ptensor_mappings_for_role(
      attrs, inputs_degrees, TensorRole::INPUT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_weight_mappings(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {

  return get_operator_to_ptensor_mappings_for_role(
      attrs, inputs_degrees, TensorRole::WEIGHT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_output_mappings(
        PCGOperatorAttrs const &op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return get_operator_to_ptensor_mappings_for_role(
      op_attrs, inputs_degrees, TensorRole::OUTPUT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_ptensor_mappings_for_role(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        TensorRole role) {

  StandardOperatorTaskGroup task_group =
      get_standard_operator_task_group(attrs, inputs_degrees);

  PCGOperatorAttrs pcg_op_attrs = pcg_op_attrs_from_compgraph_op_attrs(attrs);

  std::set<TensorSlotName> slot_names_for_role =
      keys(filter_values(get_tensor_roles(pcg_op_attrs),
                         [&](TensorRole r) -> bool { return r == role; }));

  return generate_map(
      slot_names_for_role,
      [&](TensorSlotName slot_name)
          -> OperatorSpaceToParallelTensorSpaceMapping {
        return standard_operator_task_group_get_operator_to_ptensor_mapping(
            task_group, slot_name);
      });
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_ptensor_mappings(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return merge_disjoint_maps(std::vector{
      get_operator_to_input_mappings(attrs, inputs_degrees),
      get_operator_to_weight_mappings(attrs, inputs_degrees),
      get_operator_to_output_mappings(attrs, inputs_degrees),
  });
}

} // namespace FlexFlow
