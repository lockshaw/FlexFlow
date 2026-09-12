#include "op-attrs/ops/reduce.h"
#include "utils/not_implemented.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "utils/containers/is_subseteq_of.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "utils/containers/filtrans.h"

namespace FlexFlow {

TensorShape reduce_get_output_shape(ReduceAttrs const &attrs,
                                    TensorShape const &input_shape)
{
  {
    std::set<ff_dim_t> input_dims_set = get_ff_dim_t_set(input_shape.dims);
    ASSERT(
      is_subseteq_of(attrs.axes.unwrap_as_set(), input_dims_set)
    );
  }

  if (attrs.keepdims) {
    return input_shape;
  } else {
    return tensor_shape_drop_dims(
      input_shape,
      [&](ff_dim_t d) -> bool {
        return contains(attrs.axes, d);
      });
  }
}

ParallelTensorDimDegrees reduce_get_output_parallel_dim_degrees(
          ReduceAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  auto only_shard_dims = [&](std::set<parallel_tensor_dim_idx_t> const &ds)
    -> std::set<ff_dim_t>
  {
    return filtrans(
      ds,
      [&](parallel_tensor_dim_idx_t const &d) -> std::optional<ff_dim_t> {
        return d.try_require_shard_dim();
      });
  };

  std::set<ff_dim_t> all_shard_dims =
    only_shard_dims(get_parallel_tensor_dim_indices(input_degrees));

  ASSERT(
    is_subseteq_of(attrs.axes.unwrap_as_set(), all_shard_dims)
  );

  std::set<ff_dim_t> nontrival_shard_dims =
    only_shard_dims(get_nontrivial_parallel_tensor_dim_indices(input_degrees));

  ASSERT(
    are_disjoint(attrs.axes.unwrap_as_set(), nontrival_shard_dims),
  );

  if (attrs.keepdims) {
    return input_degrees;
  } else {
    return parallel_dim_degrees_drop_shard_dims(
      input_degrees,
      [&](ff_dim_t d) -> bool {
        return contains(attrs.axes, d);
      });
  }
}

ParallelTensorShape reduce_get_output_parallel_shape(ReduceAttrs const &attrs,
                                                     ParallelTensorShape const &input_shape)
{
  TensorShape output_shape =
      reduce_get_output_shape(attrs, get_reduced_shape(input_shape));

  ParallelTensorDimDegrees output_degrees =
      reduce_get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape));

  return lift_to_parallel_with_degrees(output_shape, output_degrees);
}

StandardOperatorTaskGroup reduce_get_task_group(
    ReduceAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) 
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorTaskSpace reduce_get_operator_task_space(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reduce_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

ShardSignatureInstance
    reduce_get_shard_signature_instance(ReduceAttrs const &attrs,
                                        ParallelTensorDimDegrees const &input_degrees) {

  StandardOperatorTaskGroup op_task_group =
    reduce_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reduce_get_operator_to_input_mapping(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reduce_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reduce_get_operator_to_output_mapping(
    ReduceAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reduce_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}


} // namespace FlexFlow
