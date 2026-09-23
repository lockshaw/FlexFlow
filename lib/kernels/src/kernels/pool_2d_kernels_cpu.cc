#include "kernels/pool_2d_kernels_cpu.h"
#include "utils/not_implemented.h"
#include "utils/containers/binary_cartesian_product.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/tensor_dims_coord.h"
#include "kernels/datatype_dispatch.h"
#include "utils/containers/sum.h"

namespace FlexFlow {

static std::set<TensorDimsCoord> get_input_coords_for_output_coord(
    TensorDimsCoord const &output_coord,
    positive_int kernel_h,
    positive_int kernel_w,
    positive_int stride_h,
    positive_int stride_w)
{
  return transform(
    binary_cartesian_product(
      set_of(nonnegative_range(kernel_h)),
      set_of(nonnegative_range(kernel_w))),
    [&](std::pair<nonnegative_int, nonnegative_int> const &p)
      -> TensorDimsCoord
    {
      nonnegative_int h_offset = p.first;
      nonnegative_int w_offset = p.second;

      relative_ff_dim_t h_dim = relative_ff_dim_t{-2};
      relative_ff_dim_t w_dim = relative_ff_dim_t{-1};

      TensorDimsCoord result = output_coord;
      tensor_dims_coord_at_rel_idx(result, h_dim) *= stride_h;
      tensor_dims_coord_at_rel_idx(result, h_dim) += h_offset;

      tensor_dims_coord_at_rel_idx(result, w_dim) *= stride_w;
      tensor_dims_coord_at_rel_idx(result, w_dim) += w_offset;

      return result;
    });
}

template <DataType DT>
struct CPUPPool2DTensorAccessor {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  Pool2DAttrs const &attrs) {
    using T = real_type_t<DT>;

    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    std::function<T(std::set<T> const &)> compute_output;
    switch (attrs.pool_type) {
      case PoolOp::MAX: {
        compute_output = [&](std::set<T> const &inputs) -> T {
          return maximum(inputs);
        };
      }
      case PoolOp::AVG: {
        compute_output = [&](std::set<T> const &inputs) -> T {
          return sum(inputs) / inputs.size();
        };
      }
      default:
        PANIC();
    };

    ASSERT(!attrs.activation.has_value());

    for (TensorDimsCoord const &output_coord :
         get_tensor_dims_coord_set(output.shape.dims)) {

      std::set<TensorDimsCoord> input_coords = get_input_coords_for_output_coord(
        output_coord,
        /*kernel_h=*/attrs.kernel_h,
        /*kernel_w=*/attrs.kernel_w,
        /*stride_h=*/attrs.stride_h,
        /*stride_w=*/attrs.stride_w);

      std::set<T> input_values = transform(input_coords,
                                           [&](TensorDimsCoord const &input_coord) -> T {
                                             if (tensor_dims_contains_coord(input.shape.dims, input_coord)) {
                                               return input.at<DT>(input_coord);
                                             } else {
                                               return 0;
                                             }
                                           });

      output.at<DT>(output_coord) = compute_output(input_values);
    }
  }
};

void pool2d_cpu_forward_kernel(Pool2DAttrs const &attrs,
                               GenericTensorAccessorR const &input,
                               GenericTensorAccessorW const &output)
{
  TensorShape correct_output_shape = pool2d_get_output_shape(attrs, input.shape);
  ASSERT(output.shape == correct_output_shape);

  DataTypeDispatch1<CPUPPool2DTensorAccessor>{}(
      input.shape.data_type, input, output, attrs);
}

void pool2d_cpu_backward_kernel(Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &output_grad,
                                GenericTensorAccessorW const &input_grad)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow::Kernels::Pool2D
