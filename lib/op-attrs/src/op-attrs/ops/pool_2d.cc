#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/ops/pool_2d/pool_2d_input_shape.h"
#include "op-attrs/ops/pool_2d/pool_2d_parallel_input_shape.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

TensorShape get_kernel_shape(Pool2DAttrs const &attrs, TensorShape const &raw_input_shape) {
  
  Pool2DInputShape input = parse_pool_2d_input_shape(raw_input_shape);

  return TensorShape{
    TensorDims{
      FFOrdered<size_t>{
        size_t_from_int(attrs.kernel_h),
        size_t_from_int(attrs.kernel_w),
      }
    },
    input.datatype,
  };
}

TensorShape get_output_shape(Pool2DAttrs const &attrs, TensorShape const &raw_input_shape) {
  Pool2DInputShape input = parse_input_shape(raw_input_shape);

  size_t out_height = (input.height + 2 * attrs.padding_h - attrs.kernel_h) / attrs.stride_h + 1;
  size_t out_width = (input.width + 2 * attrs.padding_w - attrs.kernel_w) / attrs.stride_w + 1;

  assert(input.channels > 0);

  return TensorShape{
    TensorDims{
      FFOrdered<size_t>{
        input.num_samples,
        input.num_channels,
        out_height,
        out_width
      }
    },
    input.datatype
  };
}

ParallelTensorShape get_kernel_shape(Pool2DAttrs const &attrs, ParallelTensorShape const &raw_input_shape) {

  Pool2DParallelInputShape input = parse_pool_2d_parallel_input_shape(raw_input_shape);

  ShardParallelDim kernel_height_dim = {size_t_from_int(attrs.kernel_h), 1};
  ShardParallelDim kernel_width_dim = {size_t_from_int(attrs.kernel_w), 1};

  int sum_degree = 1;
  int discard_copy_degree = input.height_dim.degree * input.width_dim.degree * input.discard_copy_reduction_degree * input.sum_reduction_degree; //unsure about this line

  ParallelTensorShape result = ParallelTensorShape{
  ParallelTensorDims{
    FFOrdered<ShardParallelDim>{
      kernel_height_dim,
      kernel_width_dim,
    },
    ReplicaParallelDimSet{
      sum_degree,
      discard_copy_degree,
      },
    },
    input.datatype,
  };

  assert (total_parallel_degree(result.dims) == total_parallel_degree(raw_input_shape.dims));

  return result;
}

ParallelTensorShape get_output_shape(Pool2DAttrs const &attrs, ParallelTensorShape const &raw_input_shape) {

  Pool2DParallelInputShape input = parse_pool_2d_parallel_input_shape(raw_input_shape);

  TensorShape unpar_output_shape = get_output_shape(attrs, get_reduced_shape(raw_input_shape));

  size_t num_samples = dim_at_idx(unpar_output_shape, ff_dim_t{0});
  size_t num_channels = dim_at_idx(unpar_output_shape, ff_dim_t{1});
  size_t height = dim_at_idx(unpar_output_shape, ff_dim_t{2});
  size_t width = dim_at_idx(unpar_output_shape, ff_dim_t{3});

  ShardParallelDim sample_dim = {num_samples, input.sample_dim.degree}; 
  ShardParallelDim channel_dim = {num_channels, input.discard_copy_reduction_degree};
  ShardParallelDim height_dim = {height, input.height_dim.degree};
  ShardParallelDim width_dim = {width, input.width_dim.degree};

  int sum_degree = input.channel_dim.degree * input.sum_reduction_degree;
  int discard_copy_degree = 1;

  ParallelTensorShape result = ParallelTensorShape{
    ParallelTensorDims{
      FFOrdered<ShardParallelDim>{
        sample_dim,
        channel_dim,
        height_dim,
        width_dim,
      },
      ReplicaParallelDimSet{
        sum_degree,
        discard_copy_degree,
      },
    },
    input.datatype,
  };
  
  assert (total_parallel_degree(result.dims) == total_parallel_degree(raw_input_shape.dims));

  return result;
}

} // namespace FlexFlow
