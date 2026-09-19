#include "op-attrs/ops/element_binary.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "utils/containers/require_same.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.h"

namespace FlexFlow {

TensorShape element_binary_get_output_shape(ElementBinaryAttrs const &attrs,
                             TensorShape const &input_lhs,
                             TensorShape const &input_rhs) {
  ASSERT(!attrs.should_broadcast_lhs && !attrs.should_broadcast_rhs,
         "ElementBinary broadcasting is currently not supported. "
         "Contact @lockshaw if you want this feature implemented.");

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(input_lhs == input_rhs, "Expected input shapes to match");

    return input_lhs;
  }
}

ParallelTensorShape element_binary_get_output_parallel_shape(ElementBinaryAttrs const &attrs,
                                     ParallelTensorShape const &input_lhs,
                                     ParallelTensorShape const &input_rhs) {
  TensorShape output_shape = element_binary_get_output_shape(
      attrs, get_reduced_shape(input_lhs), get_reduced_shape(input_rhs));

  ParallelTensorDimDegrees output_degrees = element_binary_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_lhs), get_parallel_degrees(input_rhs));

  return lift_to_parallel_with_degrees(output_shape, output_degrees);
}

ParallelTensorDimDegrees element_binary_get_output_parallel_dim_degrees(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {
  ASSERT(!attrs.should_broadcast_lhs && !attrs.should_broadcast_rhs,
         "ElementBinary broadcasting is currently not supported. "
         "Contact @lockshaw if you want this feature implemented.");

  ASSERT(lhs_input_degrees == rhs_input_degrees);

  if (attrs.should_broadcast_lhs) {
    NOT_IMPLEMENTED();
  } else if (attrs.should_broadcast_rhs) {
    NOT_IMPLEMENTED();
  } else {
    ASSERT(lhs_input_degrees == rhs_input_degrees,
           "Expected input degrees to match");

    switch (attrs.type) {
      case OperatorType::EW_ADD: {
        ASSERT(
            lhs_input_degrees.discard_copy_degree.value == 1,
            "Elementwise Add expected discard copy degree of inputs to be 1");

        break;
      }
      case OperatorType::EW_SUB:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MUL:
        NOT_IMPLEMENTED();
      case OperatorType::EW_DIV:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MAX:
        NOT_IMPLEMENTED();
      case OperatorType::EW_MIN:
        NOT_IMPLEMENTED();
      default:
        PANIC("Unexpected element-wise binary operator", attrs.type);
    }

    return lhs_input_degrees;
  }
}

StandardOperatorTaskGroup element_binary_get_task_group(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    element_binary_get_output_parallel_dim_degrees(attrs, lhs_input_degrees, rhs_input_degrees);

  return StandardOperatorTaskGroup{
    filtrans(
      binary_cartesian_product(
        get_parallel_tensor_space_coordinates(lhs_input_degrees),
        get_parallel_tensor_space_coordinates(rhs_input_degrees)),
      [&](std::pair<ParallelTensorSpaceCoordinate, ParallelTensorSpaceCoordinate> const &coords)
        -> std::optional<AbstractedOperatorAtomicTaskShardBinding>
      {
        ParallelTensorSpaceCoordinate lhs_input_coord = coords.first;
        ParallelTensorSpaceCoordinate rhs_input_coord = coords.second;

        if (lhs_input_coord != rhs_input_coord) {
          retrun std::nullopt;
        }

        ParallelTensorSpaceCoordinate output_coord = require_same(lhs_input_coord, rhs_input_coord);

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_corods=*/{
            {
              TensorSlotName::LHS_INPUT,
              lhs_input_coord,
            },
            {
              TensorSlotName::RHS_INPUT,
              rhs_input_coord,
            },
            {
              TensorSlotName::OUTPUT,
              output_coord,
            },
          },
          /*task_coord=*/task_coord_matching_parallel_tensor_space_coordinate(output_coord,
                                                                              output_degrees),
        };
      }),
  };
}

ShardSignatureInstance
    element_binary_get_shard_signature_instance(
          ElementBinaryAttrs const &attrs,
          ParallelTensorDimDegrees const &lhs_input_degrees,
          ParallelTensorDimDegrees const &rhs_input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    element_binary_get_task_group(attrs, lhs_input_degrees, rhs_input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace
    element_binary_get_operator_task_space(ElementBinaryAttrs const &attrs,
                            ParallelTensorDimDegrees const &lhs_input_degrees,
                            ParallelTensorDimDegrees const &rhs_input_degrees) {

  StandardOperatorTaskGroup op_task_group =
    element_binary_get_task_group(attrs, lhs_input_degrees, rhs_input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping element_binary_get_operator_to_lhs_input_mapping(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {

  StandardOperatorTaskGroup op_task_group =
    element_binary_get_task_group(attrs, lhs_input_degrees, rhs_input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::LHS_INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping element_binary_get_operator_to_rhs_input_mapping(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {

  StandardOperatorTaskGroup op_task_group =
    element_binary_get_task_group(attrs, lhs_input_degrees, rhs_input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::RHS_INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping element_binary_get_operator_to_output_mapping(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees) {

  StandardOperatorTaskGroup op_task_group =
    element_binary_get_task_group(attrs, lhs_input_degrees, rhs_input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

} // namespace FlexFlow
