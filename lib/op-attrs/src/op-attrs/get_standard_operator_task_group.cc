#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/linear.h"
#include "utils/containers/require_two_keys.h"
#include "utils/containers/require_only_key.h"
#include "op-attrs/ops/batch_matmul.h"
#include "utils/overload.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/upsample.h"
#include "op-attrs/ops/split.h"
#include "op-attrs/ops/concat.h"
#include "op-attrs/tensor_slot_name.h"
#include "utils/containers/slice.h"
#include "op-attrs/ops/reshape.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/weight.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/attention.h"
#include "utils/containers/require_three_keys.h"

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

StandardOperatorTaskGroup get_standard_operator_task_group(
    ComputationGraphOpAttrs const &op_attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &input_dim_degree_binding)
{
  return op_attrs.visit<StandardOperatorTaskGroup>(overload {
    [&](MultiHeadAttentionAttrs const &attrs) -> StandardOperatorTaskGroup {
      auto [input_q, input_k, input_v] =
        require_three_keys(input_dim_degree_binding,
                           TensorSlotName::QUERY, TensorSlotName::KEY, TensorSlotName::VALUE);

      return attention_get_task_group(attrs, input_q, input_k, input_v);
    },
    [&](BatchMatmulAttrs const &attrs) -> StandardOperatorTaskGroup {
      auto [lhs_input_dim_degrees, rhs_input_dim_degrees] =
        require_two_keys(input_dim_degree_binding, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

      return batch_matmul_get_task_group(attrs, lhs_input_dim_degrees, rhs_input_dim_degrees);
    },
    [&](CastAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return cast_get_task_group(attrs, input_dim_degrees);
    },
    [&](ConcatAttrs const &attrs) -> StandardOperatorTaskGroup {
      std::vector<ParallelTensorDimDegrees> inputs_dim_degrees = require_only_slots_sequence(
          input_dim_degree_binding, get_variadic_inputs_slot_name_sequence());

      return concat_get_task_group(attrs, inputs_dim_degrees);
    },
    [&](Conv2DAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return conv2d_get_task_group(attrs, input_dim_degrees);
    },
    [&](DropoutAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return dropout_get_task_group(attrs, input_dim_degrees);
    },
    [&](ElementBinaryAttrs const &attrs) -> StandardOperatorTaskGroup {
      auto [lhs_input_dim_degrees, rhs_input_dim_degrees] =
        require_two_keys(input_dim_degree_binding, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

      return element_binary_get_task_group(attrs, lhs_input_dim_degrees, rhs_input_dim_degrees);
    },
    [&](ElementUnaryAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return element_unary_get_task_group(attrs, input_dim_degrees);
    },
    [&](EmbeddingAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return embedding_get_task_group(attrs, input_dim_degrees);
    },
    [&](FlatAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return flat_get_task_group(attrs, input_dim_degrees);
    },
    [&](InputAttrs const &attrs) -> StandardOperatorTaskGroup {
      ASSERT(input_dim_degree_binding.size() == 0);

      return input_get_task_group(attrs);
    },
    [&](LinearAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return linear_get_task_group(attrs, input_dim_degrees);
    },
    [&](Pool2DAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return pool2d_get_task_group(attrs, input_dim_degrees);
    },
    [&](ReshapeAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return reshape_get_task_group(attrs, input_dim_degrees);
    },
    [&](SoftmaxAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return softmax_get_task_group(attrs, input_dim_degrees);
    },
    [&](SplitAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return split_get_task_group(attrs, input_dim_degrees);
    },
    [&](TransposeAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return transpose_get_task_group(attrs, input_dim_degrees);
    },
    [&](UpsampleAttrs const &attrs) -> StandardOperatorTaskGroup {
      ParallelTensorDimDegrees input_dim_degrees =
        require_only_key(input_dim_degree_binding, TensorSlotName::INPUT);

      return upsample_get_task_group(attrs, input_dim_degrees);
    },
    [&](WeightAttrs const &attrs) -> StandardOperatorTaskGroup {
      ASSERT(input_dim_degree_binding.size() == 0);

      return weight_get_task_group(attrs);
    },
    [&](auto const &) -> StandardOperatorTaskGroup {
      PANIC();
    }
  });
}

} // namespace FlexFlow
