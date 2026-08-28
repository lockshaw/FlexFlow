#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/weight.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_two_keys.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>
#include "utils/containers/require_three_keys.h"
#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/concat.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/ops/gather.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/noop.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/reduce.h"
#include "op-attrs/ops/reverse.h"
#include "op-attrs/ops/reshape.h"
#include "op-attrs/ops/split.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/topk.h"
#include "op-attrs/ops/upsample.h"
#include "utils/containers/slice.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/tensor_slot_name.h"

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

OperatorTaskSpace get_operator_task_space(
    PCGOperatorAttrs const &attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &inputs_degrees) {
  return attrs.visit<OperatorTaskSpace>(overload{
      [&](BatchNormAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return batch_norm_get_operator_task_space(attrs, input);
      },
      [&](BatchMatmulAttrs const &attrs) -> OperatorTaskSpace {
          auto [lhs, rhs] = require_two_keys(
              inputs_degrees, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

        return batch_matmul_get_operator_task_space(attrs, lhs, rhs);
      },
      [&](BroadcastAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return broadcast_get_operator_task_space(attrs, input);
      },
      [&](CastAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return cast_get_operator_task_space(attrs, input);
      },
      [&](CombineAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return combine_get_operator_task_space(attrs, input);
      },

      [&](ConcatAttrs const &attrs) -> OperatorTaskSpace {
        std::vector<ParallelTensorDimDegrees> inputs
          = require_only_slots_sequence(inputs_degrees,
                                        get_variadic_inputs_slot_name_sequence());

        return concat_get_operator_task_space(attrs, inputs);
      },
      [&](Conv2DAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return conv2d_get_operator_task_space(attrs, input);
      },
      [&](DropoutAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return dropout_get_operator_task_space(attrs, input);
      },
      [&](ElementBinaryAttrs const &attrs) -> OperatorTaskSpace {
        auto [lhs, rhs] = require_two_keys(inputs_degrees,
                                           TensorSlotName::LHS_INPUT,
                                           TensorSlotName::RHS_INPUT);

        return element_binary_get_operator_task_space(
            /*attrs=*/attrs,
            /*lhs_input_degrees=*/lhs,
            /*rhs_input_degrees=*/rhs);
      },
      [&](ElementUnaryAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return element_unary_get_operator_task_space(attrs, input);
      },
      [&](EmbeddingAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return embedding_get_operator_task_space(attrs, input);
      },
      [&](FlatAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return flat_get_operator_task_space(attrs, input);
      },
      [&](GatherAttrs const &attrs) -> OperatorTaskSpace {
        auto [input_degrees, index_degrees] =
            require_two_keys(inputs_degrees,
                             TensorSlotName::INPUT,
                             TensorSlotName::INDEX);

        return gather_get_operator_task_space(attrs, input_degrees, index_degrees);
      },
      [&](InputAttrs const &attrs) -> OperatorTaskSpace {
        ASSERT(inputs_degrees.size() == 0);

        return input_get_operator_task_space(attrs);
      },
      [&](LayerNormAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return layer_norm_get_operator_task_space(attrs, input);
      },
      [&](LinearAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return linear_get_operator_task_space(attrs, input);
      },
      [&](MultiHeadAttentionAttrs const &attrs) -> OperatorTaskSpace {
        auto [query, key, value] =
            require_three_keys(inputs_degrees,
                               TensorSlotName::QUERY,
                               TensorSlotName::KEY,
                               TensorSlotName::VALUE);

        return attention_get_operator_task_space(attrs, query, key, value);
      },
      [&](NoopAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return noop_get_operator_task_space(attrs, input);
      },
      [&](Pool2DAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return pool2d_get_operator_task_space(attrs, input);
      },
      [&](ReduceAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return reduce_get_operator_task_space(attrs, input);
      },
      [&](ReductionAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return reduction_get_operator_task_space(attrs, input);
      },
      [&](RepartitionAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return repartition_get_operator_task_space(attrs, input);
      },
      [&](ReplicateAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return replicate_get_operator_task_space(attrs, input);
      },
      [&](ReverseAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return reverse_get_operator_task_space(attrs, input);
      },
      [&](ReshapeAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return reshape_get_operator_task_space(attrs, input);
      },
      [&](SplitAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return split_get_operator_task_space(attrs, input);
      },
      [&](SoftmaxAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return softmax_get_operator_task_space(attrs, input);
      },
      [&](TopKAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return topk_get_operator_task_space(attrs, input);
      },
      [&](TransposeAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return transpose_get_operator_task_space(attrs, input);
      },
      [&](UpsampleAttrs const &attrs) -> OperatorTaskSpace {
        ParallelTensorDimDegrees input =
            require_only_key(inputs_degrees, TensorSlotName::INPUT);

        return upsample_get_operator_task_space(attrs, input);
      },
      [&](WeightAttrs const &attrs) -> OperatorTaskSpace {
        ASSERT(inputs_degrees.size() == 0);

        return weight_get_operator_task_space(attrs);
      },
  });
}

} // namespace FlexFlow
