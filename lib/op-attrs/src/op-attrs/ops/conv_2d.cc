#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/tensor_dims.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include <libassert/assert.hpp>
#include "utils/not_implemented.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "utils/containers/require_same.h"
#include "utils/orthotope/bounded_component.h"
#include "utils/orthotope/orthotope_bounded_coord.h"
#include "utils/optional.h"
#include "op-attrs/task_space_coordinate.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_conv2d_incoming_tensor_roles(Conv2DAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
      {TensorSlotName::FILTER, IncomingTensorRole::WEIGHT},
  };

  if (attrs.use_bias) {
    result[TensorSlotName::BIAS] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

std::set<TensorSlotName> conv2d_get_slots(Conv2DAttrs const &attrs) {
  std::set<TensorSlotName> result = {
    TensorSlotName::INPUT,
    TensorSlotName::FILTER,
    TensorSlotName::OUTPUT,
  };

  if (attrs.use_bias) {
    result.insert(TensorSlotName::BIAS);
  }

  return result;
}

TensorShape conv2d_get_kernel_shape(Conv2DAttrs const &attrs,
                                    TensorShape const &raw_input_shape) {
  ASSERT(get_num_dims(raw_input_shape.dims) == 4);

  positive_int input_n = dim_at_idx(raw_input_shape.dims, ff_dim_t{0_n});
  positive_int input_c = dim_at_idx(raw_input_shape.dims, ff_dim_t{1_n});
  positive_int input_h = dim_at_idx(raw_input_shape.dims, ff_dim_t{2_n});
  positive_int input_w = dim_at_idx(raw_input_shape.dims, ff_dim_t{3_n});

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          attrs.out_channels,
          positive_int{input_c / attrs.groups},
          attrs.kernel_h,
          attrs.kernel_w,
      }},
      raw_input_shape.data_type,
  };
}

TensorShape conv2d_get_bias_shape(Conv2DAttrs const &attrs,
                                  TensorShape const &raw_input_shape) {
  ASSERT(get_num_dims(raw_input_shape.dims) == 4);

  positive_int input_n = dim_at_idx(raw_input_shape.dims, ff_dim_t{0_n});
  positive_int input_c = dim_at_idx(raw_input_shape.dims, ff_dim_t{1_n});
  positive_int input_h = dim_at_idx(raw_input_shape.dims, ff_dim_t{2_n});
  positive_int input_w = dim_at_idx(raw_input_shape.dims, ff_dim_t{3_n});

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels},
      },
      raw_input_shape.data_type,
  };
}

TensorShape conv2d_get_output_shape(Conv2DAttrs const &attrs,
                                    TensorShape const &raw_input_shape) {

  positive_int input_n = dim_at_idx(raw_input_shape.dims, ff_dim_t{0_n});
  positive_int input_c = dim_at_idx(raw_input_shape.dims, ff_dim_t{1_n});
  positive_int input_h = dim_at_idx(raw_input_shape.dims, ff_dim_t{2_n});
  positive_int input_w = dim_at_idx(raw_input_shape.dims, ff_dim_t{3_n});

  auto calculate_output_dim_size = [](positive_int input_size,
                                      nonnegative_int padding_size,
                                      positive_int kernel_size,
                                      positive_int stride) -> positive_int {
    int input_size_raw = input_size.int_from_positive_int();
    int padding_raw = padding_size.unwrap_nonnegative();
    int kernel_size_raw = kernel_size.int_from_positive_int();
    int stride_raw = stride.int_from_positive_int();

    return positive_int{
        (input_size_raw + (2 * padding_raw) - kernel_size_raw) / stride_raw + 1,
    };
  };

  positive_int out_height =
      calculate_output_dim_size(/*input_size=*/input_h,
                                /*padding_size=*/attrs.padding_h,
                                /*kernel_size=*/attrs.kernel_h,
                                /*stride_size=*/attrs.stride_h);
  positive_int out_width =
      calculate_output_dim_size(/*input_size=*/input_w,
                                /*padding_size=*/attrs.padding_w,
                                /*kernel_size=*/attrs.kernel_w,
                                /*stride_size=*/attrs.stride_w);

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              input_n,
              attrs.out_channels,
              out_height,
              out_width,
          },
      },
      raw_input_shape.data_type,
  };
}

std::map<TensorSlotName, TensorShape>
    conv2d_get_weight_shapes(Conv2DAttrs const &attrs,
                             TensorShape const &input_shape) {
  std::map<TensorSlotName, TensorShape> weight_shapes = {
      {
          TensorSlotName::FILTER,
          conv2d_get_kernel_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({
        TensorSlotName::BIAS,
        conv2d_get_bias_shape(attrs, input_shape),
    });
  }

  return weight_shapes;
}

ParallelTensorDimDegrees conv2d_get_kernel_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(input_dim_degrees) ==
         num_ptensor_shard_dims_t{4_n});

  positive_int input_sum_degree = input_dim_degrees.sum_degree.value;
  positive_int input_discard_copy_degree =
      input_dim_degrees.discard_copy_degree.value;
  positive_int input_n_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{0_n});
  positive_int input_c_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{1_n});
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{2_n}) == 1);
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{3_n}) == 1);

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      input_n_degree * input_sum_degree,
  };

  if (input_c_degree % attrs.groups == 0) {
    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * attrs.groups,
        positive_int{input_c_degree / attrs.groups},
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else if (attrs.groups % input_c_degree == 0) {
    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * input_c_degree,
        1_p,
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else {
    PANIC("input_channel_degree and group count are not compatible",
          input_c_degree,
          attrs.groups);
  }
}

ParallelTensorDimDegrees conv2d_get_bias_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(input_dim_degrees) ==
         num_ptensor_shard_dims_t{4_n});

  positive_int input_sum_degree = input_dim_degrees.sum_degree.value;
  positive_int input_discard_copy_degree =
      input_dim_degrees.discard_copy_degree.value;
  positive_int input_n_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{0_n});
  positive_int input_c_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{1_n});
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{2_n}) == 1);
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{3_n}) == 1);

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{input_n_degree};

  if (input_c_degree % attrs.groups == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree * positive_int{input_c_degree / attrs.groups},
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * attrs.groups,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else if (attrs.groups % input_c_degree == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree,
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * input_c_degree,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else {
    PANIC("input_channel_degree and group count are not compatible",
          input_c_degree,
          attrs.groups);
  }
}

ParallelTensorDimDegrees conv2d_get_output_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(input_dim_degrees) ==
         num_ptensor_shard_dims_t{4_n});

  positive_int input_sum_degree = input_dim_degrees.sum_degree.value;
  positive_int input_discard_copy_degree =
      input_dim_degrees.discard_copy_degree.value;
  positive_int input_n_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{0_n});
  positive_int input_c_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{1_n});
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{2_n}) == 1);
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{3_n}) == 1);

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};

  if (input_c_degree % attrs.groups == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree * positive_int{input_c_degree / attrs.groups},
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_n_degree,
        input_discard_copy_degree * attrs.groups,
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else if (attrs.groups % input_c_degree == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree,
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_n_degree,
        input_discard_copy_degree * input_c_degree,
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else {
    PANIC("input_channel_degree and group count are not compatible",
          input_c_degree,
          attrs.groups);
  }
}


std::map<TensorSlotName, ParallelTensorDimDegrees>
    conv2d_get_weight_parallel_dim_degrees(Conv2DAttrs const &attrs,
                             ParallelTensorDimDegrees const &input_degrees)
{
  std::map<TensorSlotName, ParallelTensorDimDegrees> weight_degrees = {
      {
          TensorSlotName::FILTER,
          conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees),
      },
  };

  if (attrs.use_bias) {
    weight_degrees.insert({
        TensorSlotName::BIAS,
        conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees),
    });
  }

  return weight_degrees;
}

StandardOperatorTaskGroup conv2d_get_task_group(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ParallelTensorDimDegrees output_degrees =
    conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

  StandardOperatorTaskGroup task_group = StandardOperatorTaskGroup{
    transform(
      get_parallel_tensor_space_coordinates(input_degrees),
      [&](ParallelTensorSpaceCoordinate const &input_coord)
        -> AbstractedOperatorAtomicTaskShardBinding
      {
        num_ptensor_shard_dims_t input_num_shard_dims =
          get_ptensor_dim_degrees_num_shard_dims(input_degrees);

        parallel_tensor_dim_idx_t input_sum_dim = sum_dim_idx();
        parallel_tensor_dim_idx_t input_discard_copy_dim = discard_copy_dim_idx();

        parallel_tensor_dim_idx_t input_batch_dim =
          shard_dim_idx(ff_dim_t{0_n});

        parallel_tensor_dim_idx_t input_channel_dim =
          shard_dim_idx(ff_dim_t{1_n});

        parallel_tensor_dim_idx_t input_height_dim =
          shard_dim_idx(ff_dim_t{2_n});

        parallel_tensor_dim_idx_t input_width_dim =
          shard_dim_idx(ff_dim_t{3_n});

        BoundedComponent data_parallelism_component =
            bounded_component_for_ptensor_dim(
              input_degrees,
              input_coord,
              input_batch_dim);

        {
          BoundedComponent output_channel_parallelism_component =
              bounded_component_for_ptensor_dim(
                input_degrees,
                input_coord,
                input_discard_copy_dim);
          ASSERT(output_channel_parallelism_component == trivial_bounded_component());
        }

        {
          BoundedComponent preexisting_sum_parallelism_component =
              bounded_component_for_ptensor_dim(
                input_degrees,
                input_coord,
                input_sum_dim);
          ASSERT(preexisting_sum_parallelism_component == trivial_bounded_component());
        }

        {
          BoundedComponent input_channel_parallelism_component =
              bounded_component_for_ptensor_dim(
                input_degrees,
                input_coord,
                input_channel_dim);
          ASSERT(input_channel_parallelism_component == trivial_bounded_component());
        }

        {
          BoundedComponent input_height_parallelism_component =
              require_same(
                bounded_component_for_ptensor_dim(
                  input_degrees,
                  input_coord,
                  input_height_dim),
                trivial_bounded_component());
          ASSERT(input_height_parallelism_component == trivial_bounded_component());
        }

        {
          BoundedComponent input_width_parallelism_component =
              require_same(
                bounded_component_for_ptensor_dim(
                  input_degrees,
                  input_coord,
                  input_width_dim),
                trivial_bounded_component());
          ASSERT(input_width_parallelism_component == trivial_bounded_component());
        }

        ParallelTensorSpaceCoordinate output_coord =
              parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                /*sum_degree=*/trivial_bounded_component(),
                /*discard_copy_degree=*/trivial_bounded_component(),
                /*shard_coords=*/make_4d_orthotope_bounded_coord(
                  data_parallelism_component,
                  trivial_bounded_component(),
                  trivial_bounded_component(),
                  trivial_bounded_component()));

        return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
            {
              TensorSlotName::INPUT,
              input_coord,
            },
            {
              TensorSlotName::FILTER,
              parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                /*sum_coord=*/trivial_bounded_component(),
                /*discard_copy_coord=*/data_parallelism_component,
                /*shard_coords=*/make_4d_orthotope_bounded_coord(
                  trivial_bounded_component(),
                  trivial_bounded_component(),
                  trivial_bounded_component(),
                  trivial_bounded_component())),
            },
            {
              TensorSlotName::BIAS,
              parallel_tensor_space_coordinate_from_bounded_orthotope_components(
                /*sum_coord=*/trivial_bounded_component(),
                /*discard_copy_coord=*/data_parallelism_component,
                /*shard_coords=*/lift_bounded_component(trivial_bounded_component())),
            },
            {
              TensorSlotName::OUTPUT,
              output_coord
            },
          },
          /*task_coord=*/task_coord_matching_parallel_tensor_space_coordinate(output_coord, output_degrees),
        };
      }),
  };

  return restrict_standard_operator_task_group_to_slots(
    task_group,
    conv2d_get_slots(attrs));
}

ShardSignatureInstance
    conv2d_get_shard_signature_instance(
          Conv2DAttrs const &attrs,
          ParallelTensorDimDegrees const &input_degrees) {
  StandardOperatorTaskGroup op_task_group =
    conv2d_get_task_group(attrs, input_degrees);

  return shard_signature_instance_from_standard_operator_task_group(op_task_group);
}

OperatorTaskSpace conv2d_get_operator_task_space(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    conv2d_get_task_group(attrs, input_degrees);

  return task_space_for_standard_operator_task_group(op_task_group);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  conv2d_get_operator_to_input_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    conv2d_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::INPUT);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  conv2d_get_operator_to_kernel_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    conv2d_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::FILTER);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  conv2d_get_operator_to_bias_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  ASSERT(attrs.use_bias);

  StandardOperatorTaskGroup op_task_group =
    conv2d_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::BIAS);
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
  conv2d_get_operator_to_output_mapping(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  StandardOperatorTaskGroup op_task_group =
    conv2d_get_task_group(attrs, input_degrees);

  return standard_operator_task_group_get_operator_to_ptensor_mapping(op_task_group, TensorSlotName::OUTPUT);
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
  conv2d_get_operator_to_parallel_tensor_mappings(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees)
{
  std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping> mappings = {
      {
          TensorSlotName::INPUT,
          conv2d_get_operator_to_input_mapping(attrs, input_degrees),
      },
      {
          TensorSlotName::FILTER,
          conv2d_get_operator_to_kernel_mapping(attrs, input_degrees),
      },
      {
          TensorSlotName::OUTPUT,
          conv2d_get_operator_to_output_mapping(attrs, input_degrees),
      },
  };

  if (attrs.use_bias) {
    mappings.insert({
        TensorSlotName::BIAS,
        conv2d_get_operator_to_bias_mapping(attrs, input_degrees),
    });
  }

  return mappings;
}

std::map<TensorSlotName, OperatorSpaceToParallelTensorSpaceBiuniqueMapping>
  conv2d_get_operator_to_parallel_tensor_mappings(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input);

ParallelTensorShape
    conv2d_get_kernel_parallel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  TensorShape unpar = conv2d_get_kernel_shape(attrs, get_reduced_shape(input));
  ParallelTensorDimDegrees degrees = conv2d_get_kernel_parallel_dim_degrees(
      attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

ParallelTensorShape
    conv2d_get_bias_parallel_shape(Conv2DAttrs const &attrs,
                                   ParallelTensorShape const &input) {
  TensorShape unpar = conv2d_get_bias_shape(attrs, get_reduced_shape(input));
  ParallelTensorDimDegrees degrees =
      conv2d_get_bias_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

ParallelTensorShape
    conv2d_get_output_parallel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {

  TensorShape unpar = conv2d_get_output_shape(attrs, get_reduced_shape(input));
  ParallelTensorDimDegrees degrees = conv2d_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

std::map<TensorSlotName, ParallelTensorShape>
    conv2d_get_weight_parallel_shapes(Conv2DAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {
  std::map<TensorSlotName, ParallelTensorShape> weight_shapes = {
      {
          TensorSlotName::FILTER,
          conv2d_get_kernel_parallel_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({
        TensorSlotName::BIAS,
        conv2d_get_bias_parallel_shape(attrs, input_shape),
    });
  }

  return weight_shapes;
}

/**
 * @brief Chosen to match pytorch implementation
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/conv.py#L178-L187
 */
std::map<TensorSlotName, InitializerAttrs> conv2d_get_initializers(
    Conv2DAttrs const &attrs,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> maybe_kernel_initializer,
    std::optional<InitializerAttrs> maybe_bias_initializer) {

  TensorShape kernel_shape = conv2d_get_kernel_shape(attrs, input_shape);

  InitializerAttrs kernel_default_initializer =
      InitializerAttrs{KaimingNormalAttrs{
          /*a=*/sqrtf(5.0),
          /*mode=*/KaimingInitializerMode::FAN_IN,
          /*nonlinearity=*/KaimingInitializerNonlinearity::LEAKY_RELU,
          /*seed=*/0,
      }};

  InitializerAttrs kernel_initializer =
      maybe_kernel_initializer.value_or(kernel_default_initializer);

  positive_int fan_in =
      calculate_fan_for_mode(kernel_shape.dims, KaimingInitializerMode::FAN_IN);

  float bound = 1 / sqrtf(static_cast<float>(fan_in.int_from_positive_int()));

  InitializerAttrs bias_default_initializer =
      InitializerAttrs{UniformInitializerAttrs{
          /*seed=*/0,
          /*min_val=*/-bound,
          /*max_val=*/bound,
      }};

  InitializerAttrs bias_initializer =
      maybe_bias_initializer.value_or(bias_default_initializer);

  if (attrs.use_bias) {
    return {
        {TensorSlotName::FILTER, kernel_initializer},
        {TensorSlotName::BIAS, bias_initializer},
    };
  } else {
    return {
        {TensorSlotName::FILTER, kernel_initializer},
    };
  }
}

} // namespace FlexFlow
