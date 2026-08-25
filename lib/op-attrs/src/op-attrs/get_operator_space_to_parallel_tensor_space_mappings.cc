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

namespace FlexFlow {

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings(
        PCGOperatorAttrs const &op_attrs,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees) {
  return op_attrs.visit<
      std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>>(
      overload{
          [&](ElementBinaryAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ASSERT(inputs_degrees.size() == 2);

            ParallelTensorDimDegrees lhs_degrees =
                inputs_degrees.at(TensorSlotName::LHS_INPUT);
            ParallelTensorDimDegrees rhs_degrees =
                inputs_degrees.at(TensorSlotName::RHS_INPUT);

            return {
                {
                    TensorSlotName::LHS_INPUT,
                    element_binary_get_operator_to_lhs_input_mapping(
                        attrs, lhs_degrees, rhs_degrees),
                },
                {
                    TensorSlotName::RHS_INPUT,
                    element_binary_get_operator_to_rhs_input_mapping(
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
                    TensorSlotName::INPUT,
                    element_unary_get_operator_to_input_mapping(attrs, input_degrees),
                },
            };
          },
          [&](InputAttrs const &) {
            ASSERT(inputs_degrees.size() == 0);

            return std::map<TensorSlotName,
                            OperatorSpaceToParallelTensorSpaceMapping>{};
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
          [&](TransposeAttrs const &attrs)
              -> std::map<TensorSlotName,
                          OperatorSpaceToParallelTensorSpaceMapping> {
            ParallelTensorDimDegrees input_degrees =
                require_only_key(inputs_degrees, TensorSlotName::INPUT);

            return {
                {
                    TensorSlotName::INPUT,
                    transpose_get_operator_to_input_mapping(attrs, input_degrees),
                },
            };
          },
          [&](WeightAttrs const &) {
            ASSERT(inputs_degrees.size() == 0);

            return std::map<TensorSlotName,
                            OperatorSpaceToParallelTensorSpaceMapping>{};
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
