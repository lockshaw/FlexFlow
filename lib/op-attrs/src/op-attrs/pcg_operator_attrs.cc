#include "op-attrs/pcg_operator_attrs.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/weight.h"
#include "utils/overload.h"

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &attrs) {
  return (attrs.has<CombineAttrs>() || attrs.has<ReductionAttrs>() ||
          attrs.has<RepartitionAttrs>() || attrs.has<ReplicateAttrs>());
}

OperatorType pcg_op_attrs_get_op_type(PCGOperatorAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

nlohmann::json pcg_op_attrs_as_dot_json(PCGOperatorAttrs const &attrs) {
  nlohmann::json result = attrs;

  return result;
}

PCGOperatorAttrs pcg_op_attrs_from_compgraph_op_attrs(
    ComputationGraphOpAttrs const &cg_attrs) {
  return cg_attrs.visit<PCGOperatorAttrs>(overload{
      [](auto const &attrs) { return PCGOperatorAttrs{attrs}; },
  });
}

void pcg_op_attrs_check_incoming_tensor_roles(
    PCGOperatorAttrs const &op_attrs,
    std::unordered_set<TensorSlotName> const &input_slots,
    std::unordered_set<TensorSlotName> const &weight_slots) {
  std::unordered_map<TensorSlotName, IncomingTensorRole> correct =
      get_incoming_tensor_roles(op_attrs);

  std::unordered_map<TensorSlotName, IncomingTensorRole> current =
      binary_merge_disjoint_maps(
          generate_map(
              input_slots,
              [](TensorSlotName) { return IncomingTensorRole::INPUT; }),
          generate_map(weight_slots, [](TensorSlotName) {
            return IncomingTensorRole::WEIGHT;
          }));

  ASSERT(correct == current,
         "check_incoming_tensor_roles found deviation in incoming tensors");
}

} // namespace FlexFlow
