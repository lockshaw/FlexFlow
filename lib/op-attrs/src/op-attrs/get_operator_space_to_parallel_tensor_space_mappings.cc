#include "op-attrs/get_operator_space_to_parallel_tensor_space_mappings.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/ops/element_binary.h"
#include "op-attrs/ops/element_unary.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/transpose.h"
#include "op-attrs/ops/weight.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/get_only.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/require_two_keys.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/overload.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/replicate.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/slice.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/batch_matmul.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/concat.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/dropout.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/flat.h"
#include "op-attrs/tensor_slot_name.h"
#include "utils/containers/restrict_keys_strict.h"
#include "op-attrs/ops/upsample.h"
#include "op-attrs/ops/topk.h"
#include "op-attrs/ops/softmax.h"
#include "op-attrs/ops/split.h"
#include "op-attrs/ops/reshape.h"
#include "op-attrs/ops/reverse.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/reduce.h"
#include "op-attrs/ops/noop.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/cast.h"

namespace FlexFlow {

template <typename T>
static std::tuple<T, T, T> require_3(std::map<TensorSlotName, T> const &v,
                                     TensorSlotName k1,
                                     TensorSlotName k2,
                                     TensorSlotName k3) {
  ASSERT(v.size() == 3);

  return {v.at(k1), v.at(k2), v.at(k3)};
}

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

static 
std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
  lift_biunique_mappings(
    std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping> const &biunique)
{
  return map_values(biunique, operator_ptensor_space_mapping_from_biunique);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings(
        PCGOperatorAttrs const &op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return op_attrs.visit<
      std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>>(
      overload{
          [&](BatchNormAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      batch_norm_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](BatchMatmulAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            auto [lhs, rhs] = require_two_keys(
                inputs_degrees, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

            return {
                {
                    TensorSlotName::LHS_INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      batch_matmul_get_operator_to_lhs_input_mapping(attrs, lhs, rhs)),
                },
                {
                    TensorSlotName::RHS_INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      batch_matmul_get_operator_to_rhs_input_mapping(attrs, lhs, rhs)),
                },
            };
          },
          [&](BroadcastAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      broadcast_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](CastAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      cast_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](CombineAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    combine_get_operator_to_input_mapping(attrs, input_degrees),
                },
            };
          },
          [&](ConcatAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            std::vector<ParallelTensorDimDegrees> inputs 
              = require_only_slots_sequence(inputs_degrees,
                                            get_variadic_inputs_slot_name_sequence());

            std::vector<OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
              concat_get_operator_to_input_mappings(attrs, inputs);

            return lift_biunique_mappings(
              map_from_keys_and_values(
                slice(get_variadic_inputs_slot_name_sequence(), 0, inputs.size()),
                concat_get_operator_to_input_mappings(attrs, inputs)));
          },
          [&](Conv2DAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            std::set<TensorSlotName> incoming_slots = 
              keys(get_conv2d_incoming_tensor_roles(attrs));

            return lift_biunique_mappings(
              restrict_keys_strict(
                conv2d_get_operator_to_parallel_tensor_mappings(attrs, input_degrees),
                incoming_slots));
          },
          [&](DropoutAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      dropout_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](ElementBinaryAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            auto [lhs, rhs] = require_two_keys(
                inputs_degrees, TensorSlotName::LHS_INPUT, TensorSlotName::RHS_INPUT);

            return {
                {
                    TensorSlotName::LHS_INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      element_binary_get_operator_to_lhs_input_mapping(
                          attrs, lhs, rhs)),
                },
                {
                    TensorSlotName::RHS_INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      element_binary_get_operator_to_rhs_input_mapping(
                          attrs, lhs, rhs)),
                },
            };
          },
          [&](ElementUnaryAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      element_unary_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](EmbeddingAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                  TensorSlotName::INPUT,
                  operator_ptensor_space_mapping_from_biunique(
                    embedding_get_operator_to_input_mapping(attrs, input_degrees)),
                },
                {
                  TensorSlotName::WEIGHT,
                  operator_ptensor_space_mapping_from_biunique(
                    embedding_get_operator_to_weights_mapping(attrs, input_degrees)),
                },
            };

          },
          [&](FlatAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      flat_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](GatherAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            auto [input_degrees, index_degrees] =
                require_two_keys(inputs_degrees, 
                                 TensorSlotName::INPUT,
                                 TensorSlotName::INDEX);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      gather_get_operator_to_input_mapping(attrs, input_degrees, index_degrees)),
                },
                {
                    TensorSlotName::INDEX,
                    operator_ptensor_space_mapping_from_biunique(
                      gather_get_operator_to_input_mapping(attrs, input_degrees, index_degrees)),
                },
            };
          },
          [&](InputAttrs const &) {
            ASSERT(inputs_degrees.size() == 0);

            return std::map<TensorSlotName,
                            OperatorSpaceToParallelTensorSpaceMapping>{};
          },
          [&](LayerNormAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      layer_norm_get_operator_to_input_mapping(attrs, input_degrees)),
                },
                {
                    TensorSlotName::GAMMA,
                    operator_ptensor_space_mapping_from_biunique(
                      layer_norm_get_operator_to_input_mapping(attrs, input_degrees)),
                },
                {
                    TensorSlotName::BETA,
                    operator_ptensor_space_mapping_from_biunique(
                      layer_norm_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](LinearAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
                result = {
                    {
                        TensorSlotName::INPUT,
                        operator_ptensor_space_mapping_from_biunique(
                          linear_get_operator_to_input_mapping(attrs,
                                                               input_degrees)),
                    },
                    {
                        TensorSlotName::WEIGHT,
                        operator_ptensor_space_mapping_from_biunique(
                          linear_get_operator_to_projection_mapping(
                              attrs, input_degrees)),
                    },
                };

            if (attrs.use_bias) {
              result.insert({
                TensorSlotName::BIAS,
                operator_ptensor_space_mapping_from_biunique(
                    linear_get_operator_to_bias_mapping(attrs, input_degrees)),
              });
            };

            return result;
          },
          [&](MultiHeadAttentionAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            auto [query, key, value] =
                require_only_key(inputs_degrees, 
                                 TensorSlotName::QUERY,
                                 TensorSlotName::KEY,
                                 TensorSlotName::VALUE);

            std::set<TensorSlotName> incoming_slots = 
              keys(get_attention_incoming_tensor_roles(attrs));

            return lift_biunique_mappings(
              restrict_keys_strict(
                attention_get_operator_to_parallel_tensor_mappings(attrs, query, key, value),
                incoming_slots));
          },
          [&](NoopAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      noop_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](Pool2DAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      pool2d_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](ReduceAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      reduce_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](ReductionAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    reduction_get_operator_to_input_mapping(attrs,
                                                              input_degrees),
                },
            };
          },
          [&](RepartitionAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    repartition_get_operator_to_input_mapping(attrs,
                                                              input_degrees),
                },
            };
          },
          [&](ReplicateAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    replicate_get_operator_to_input_mapping(attrs,
                                                              input_degrees),
                },
            };
          },
          [&](ReverseAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      reverse_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](ReshapeAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      reshape_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](SplitAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      split_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](SoftmaxAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      softmax_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](TopKAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      topk_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](TransposeAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      transpose_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](UpsampleAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      upsample_get_operator_to_input_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](WeightAttrs const &) {
            ASSERT(inputs_degrees.size() == 0);

            return std::map<TensorSlotName,
                            OperatorSpaceToParallelTensorSpaceMapping>{};
          },
      });
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings_for_role(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        IncomingTensorRole incoming_tensor_role) {

  std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
      incoming_mappings =
          get_operator_to_incoming_mappings(attrs, inputs_degrees);

  std::map<TensorSlotName, IncomingTensorRole> incoming_tensor_roles =
      get_incoming_tensor_roles(attrs);

  return filtermap_values(
      zip_values_strict(incoming_mappings, incoming_tensor_roles),
      [&](std::pair<OperatorSpaceToParallelTensorSpaceMapping,
                    IncomingTensorRole> const &p)
          -> std::optional<OperatorSpaceToParallelTensorSpaceMapping> {
        auto const &[mapping, role] = p;

        if (role == incoming_tensor_role) {
          return mapping;
        } else {
          return std::nullopt;
        }
      });
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_input_mappings(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return get_operator_to_incoming_mappings_for_role(
      attrs, inputs_degrees, IncomingTensorRole::INPUT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_weight_mappings(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {

  return get_operator_to_incoming_mappings_for_role(
      attrs, inputs_degrees, IncomingTensorRole::WEIGHT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_output_mappings(
        PCGOperatorAttrs const &op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {

  return op_attrs.visit<
      std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>>(
      overload{
          [&](ElementBinaryAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            auto [lhs_degrees, rhs_degrees] =
                require_two_keys(inputs_degrees,
                                 TensorSlotName::LHS_INPUT,
                                 TensorSlotName::RHS_INPUT);

            return {
                {
                    TensorSlotName::OUTPUT,
                    element_binary_get_operator_to_output_mapping(
                        attrs, lhs_degrees, rhs_degrees),
                },
            };
          },
          [&](ElementUnaryAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::OUTPUT,
                    element_unary_get_operator_to_output_mapping(attrs, input_degrees),
                },
            };
          },
          [&](LinearAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::OUTPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      linear_get_operator_to_output_mapping(attrs, input_degrees)),
                },
            };
          },
          [&](InputAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ASSERT(inputs_degrees.size() == 0);

            return {
                {
                    TensorSlotName::OUTPUT,
                    input_get_operator_to_output_mapping(attrs),
                },
            };
          },
          [&](ReductionAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::OUTPUT,
                    reduction_get_operator_to_output_mapping(attrs,
                                                               input_degrees),
                },
            };
          },
          [&](RepartitionAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::OUTPUT,
                    operator_ptensor_space_mapping_from_biunique(
                      repartition_get_operator_to_output_mapping(attrs,
                                                                 input_degrees)),
                },
            };
          },
          [&](ReplicateAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::OUTPUT,
                    replicate_get_operator_to_output_mapping(attrs,
                                                               input_degrees),
                },
            };
          },
          [&](TransposeAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::OUTPUT,
                    transpose_get_operator_to_output_mapping(attrs, input_degrees),
                },
            };
          },
          [&](WeightAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ASSERT(inputs_degrees.size() == 0);

            return {
                {
                    TensorSlotName::OUTPUT,
                    weight_get_operator_to_output_mapping(attrs),
                },
            };
          },
          [](auto const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            PANIC("Missing implmentation of get_operator_to_input_mappings",
                  attrs);
          },
      });
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_ptensor_mappings_for_role(
        PCGOperatorAttrs const &attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        TensorRole role) {
  switch (role) {
    case TensorRole::INPUT:
      return get_operator_to_input_mappings(attrs, inputs_degrees);
    case TensorRole::WEIGHT:
      return get_operator_to_weight_mappings(attrs, inputs_degrees);
    case TensorRole::OUTPUT:
      return get_operator_to_output_mappings(attrs, inputs_degrees);
    default:
      PANIC("Unhandled TensorRole", role);
  }
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
