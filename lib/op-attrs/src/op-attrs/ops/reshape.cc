#include "op-attrs/ops/reshape.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/product.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/containers/take_while.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/containers/set_union.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/ff_ordered/ff_ordered_concat.h"

namespace FlexFlow {

TensorShape reshape_get_output_shape(ReshapeAttrs const &attrs,
                                     TensorShape const &input_shape) {

  TensorDims input_dims = input_shape.dims;
  ff_dim_t first_nonleading_dim = ff_dim_t{
    get_num_dims(input_dims).nonnegative_int_from_num_tensor_dims(),
  };

  {
    TensorDims reduced_input_dims = 
      slice_tensor_dims(input_dims, first_nonleading_dim, std::nullopt);

    ASSERT(reduced_input_dims == attrs.core_input_dims);
  }

  TensorDims leading_dims = slice_tensor_dims(
    input_dims, ff_dim_t{0_n}, first_nonleading_dim);

  TensorDims output_dims = TensorDims{
    ff_ordered_concat(leading_dims.ff_ordered, attrs.core_output_dims.ff_ordered),
  };

  return TensorShape{
    /*dims=*/output_dims,
    /*data_type=*/input_shape.data_type,
  };
}

static
std::set<parallel_tensor_dim_idx_t> get_dims_allowed_parallel(ReshapeAttrs const &attrs)
{
  num_tensor_dims_t max_dims = std::max(
    get_num_dims(attrs.core_input_dims),
    get_num_dims(attrs.core_output_dims));

  std::set<ff_dim_t>
    leading_dims = set_of(
      take_while(
        ff_dim_range(max_dims.nonnegative_int_from_num_tensor_dims()),
        [&](ff_dim_t d) -> bool {
          positive_int input_dim_size = dim_at_idx(attrs.core_input_dims, d);
          positive_int output_dim_size = dim_at_idx(attrs.core_output_dims, d);

          return input_dim_size == output_dim_size;
        }));

  std::set<parallel_tensor_dim_idx_t> dims_allowed_parallel =
    set_union(
      transform(leading_dims, 
                [&](ff_dim_t d) -> parallel_tensor_dim_idx_t {
                  return shard_dim_idx(d); 
                }),
      std::set{sum_dim_idx(), discard_copy_dim_idx()});

  return dims_allowed_parallel;
}

ParallelTensorDimDegrees reshape_get_output_parallel_dim_degrees(
    ReshapeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {

  std::set<parallel_tensor_dim_idx_t> dims_allowed_parallel =
    get_dims_allowed_parallel(attrs);

  std::set<parallel_tensor_dim_idx_t> dims_not_allowed_parallel =
    set_minus(get_parallel_tensor_dim_indices(input_dim_degrees),
              dims_allowed_parallel);

  ASSERT(
    all_of(dims_not_allowed_parallel,
           [&](parallel_tensor_dim_idx_t d) -> bool {
             return get_degree_for_parallel_tensor_dim_idx(input_dim_degrees, d) == 1;
           })
  );

  return input_dim_degrees;
}

ParallelTensorShape
    reshape_get_output_parallel_shape(ReshapeAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      reshape_get_output_shape(attrs, get_reduced_shape(input_shape));
  ParallelTensorDimDegrees degrees = reshape_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input_shape));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

StandardOperatorTaskGroup reshape_get_task_group(
    ReshapeAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    reshape_get_output_parallel_dim_degrees(attrs, input_degrees);

  return StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        ParallelTensorSpaceCoordinate output_coord = input_coord;

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/{
            {
              TensorSlotName::INPUT,
              input_coord,
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
    reshape_get_shard_signature_instance(
          ReshapeAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace reshape_get_operator_task_space(
    ReshapeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reshape_get_operator_to_input_mapping(
    ReshapeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping reshape_get_operator_to_output_mapping(
    ReshapeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    reshape_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

} // namespace FlexFlow
