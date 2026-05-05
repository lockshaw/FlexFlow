#include "op-attrs/ops/repartition.h"
#include <libassert/assert.hpp>
#include "op-attrs/operator_task_space.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(RepartitionAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  ParallelTensorShape output_shape = input_shape;
  output_shape.dims.shard_dims
      .at(relative_ff_dim_t_from_ff_dim_t(attrs.repartition_dim))
      .degree *= attrs.repartition_degree;
  return output_shape;
}

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    RepartitionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees output_degrees = input_degrees;
  output_degrees.shard_degrees
      .at(relative_ff_dim_t_from_ff_dim_t(attrs.repartition_dim))
      *= attrs.repartition_degree;
  return output_degrees;
}

OperatorTaskSpace
    get_operator_task_space(RepartitionAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees = get_output_parallel_dim_degrees(
      attrs, input_degrees);

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    RepartitionAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) 
{
  OperatorTaskSpace op_task_space = get_operator_task_space(attrs, input_degrees);

  DimDomain<operator_task_space_dim_idx_t> op_task_dim_domain =
    lift_minimal_dim_domain(minmial_dim_domain_from_operator_task_space(op_task_space));

  OperatorTaskSpace  get_identity_mapping(get_operator_task_space(attrs, input_degrees),
                              output_degrees);

  std::unordered_set<
    std::pair<
      DimCoord<operator_task_space_dim_idx_t>,
      DimCoord<parallel_tensor_dim_idx_t>
    >
  > dim_domain_coord_relation = 
    transform(
      get_coords_in_dim_domain(op_task_dim_domain),
      [](DimCoord<operator_task_space_dim_idx_t> const &c) {
        return 
      });

  DimDomainHemiuniqueMapping<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> result =   
    DimDomainHemiuniqueMapping<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>{
      OneToMany<
        DimCoord<operator_task_space_dim_idx_t>, 
        DimCoord<parallel_tensor_dim_idx_t>>
      {
      },
    };
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    RepartitionAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) 
{
  ParallelTensorDimDegrees output_degrees =
      get_output_parallel_dim_degrees(attrs, input_degrees);

  return get_identity_mapping(get_operator_task_space(attrs, input_degrees),
                              output_degrees);
}


} // namespace FlexFlow
