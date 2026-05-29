#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/weight.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_two_keys.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/reduction.h"

namespace FlexFlow {

OperatorTaskSpace get_operator_task_space(
    PCGOperatorAttrs const &attrs,
    std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
        &inputs_degrees) {
  return attrs.visit<OperatorTaskSpace>(overload{
      [&](CombineAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return get_operator_task_space(attrs, input);
      },
      [&](ElementUnaryAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return get_operator_task_space(attrs, input);
      },
      [&](ElementBinaryAttrs const &attrs) {
        auto [lhs, rhs] = require_two_keys(inputs_degrees,
                                           TensorSlotName::LHS_INPUT,
                                           TensorSlotName::RHS_INPUT);

        return get_operator_task_space(
            /*attrs=*/attrs,
            /*lhs_input_degrees=*/lhs,
            /*rhs_input_degrees=*/rhs);
      },
      [&](LinearAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return linear_get_operator_task_space(attrs, input);
      },
      [&](InputAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 0);

        return get_operator_task_space(attrs);
      },
      [&](ReductionAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return get_operator_task_space(attrs, input);
      },
      [&](RepartitionAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return repartition_get_operator_task_space(attrs, input);
      },
      [&](ReplicateAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return replicate_get_operator_task_space(attrs, input);
      },
      [&](TransposeAttrs const &attrs) {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return get_operator_task_space(attrs, input);
      },
      [&](WeightAttrs const &attrs) {
        ASSERT(inputs_degrees.size() == 0);

        return get_operator_task_space(attrs);
      },
      [](auto const &attrs) -> OperatorTaskSpace {
        PANIC("Missing implmentation of get_operator_task_space", attrs);
      },
  });
}

} // namespace FlexFlow
