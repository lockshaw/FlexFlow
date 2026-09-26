#include "kernels/layer_norm_kernels_cpu.h"
#include "utils/not_implemented.h"
#include "utils/containers/compute_mean.h"
#include "kernels/reduce_tensor_accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "utils/containers/compute_variance.h"
#include "op-attrs/tensor_dims.h"
#include "kernels/tensor_accessor_promote_dims.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "utils/optional.h"
#include "kernels/accessor.h"
#include "utils/overload.h"

namespace FlexFlow {

void layer_norm_cpu_forward_kernel(LayerNormAttrs const &attrs,
                                   GenericTensorAccessorR const &input,
                                   GenericTensorAccessorW const &output,
                                   std::optional<GenericTensorAccessorR> const &gamma,
                                   std::optional<GenericTensorAccessorR> const &beta) {
  ASSERT(attrs.elementwise_affine == gamma.has_value());
  ASSERT(attrs.elementwise_affine == beta.has_value());

  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  GenericTensorAccessorW mean =
    reduce_tensor_accessor_in_dims(
      input,
      attrs.axes,
      cpu_allocator,
      overload {
        [&](std::vector<float> const &input_values) -> float {
          return compute_mean(input_values);
        },
        [](auto const &x) -> get_element_type_t<decltype(x)> {
          PANIC();
        },
      });

  GenericTensorAccessorW variance =
    reduce_tensor_accessor_in_dims(
      input,
      attrs.axes,
      cpu_allocator,
      overload {
        [&](std::vector<float> const &input_values) -> float {
          return compute_variance(input_values);
        },
        [](auto const &x) -> get_element_type_t<decltype(x)> {
          PANIC();
        },
      });

  TensorDims promoted_mean_dims = 
    tensor_dims_transform_with_idx(
      input.shape.dims,
      [&](ff_dim_t dim_idx, positive_int dim_size) -> positive_int {
        if (contains(attrs.axes, dim_idx)) {
          return 1_p;
        } else {
          return dim_size;
        }
      });
  TensorDims promoted_variance_dims = promoted_mean_dims;

  GenericTensorAccessorR promoted_mean = 
    tensor_accessor_promote_dims(mean, promoted_mean_dims);

  GenericTensorAccessorR promoted_variance = 
    tensor_accessor_promote_dims(variance, promoted_variance_dims);

  auto broadcast = [&](GenericTensorAccessorR const &t) -> GenericTensorAccessorW {
    return tensor_accessor_broadcast(
      t,
      input.shape.dims,
      cpu_allocator);
  };

  GenericTensorAccessorW broadcasted_mean = broadcast(promoted_mean);
  GenericTensorAccessorW broadcasted_variance = broadcast(promoted_variance);

  GenericTensorAccessorW numerator = 
      tensor_accessor_elementwise_subtract(input, broadcasted_mean, cpu_allocator);

  GenericTensorAccessorW denominator = 
    tensor_accessor_sqrt(
      tensor_accessor_add_constant(broadcasted_variance, attrs.eps, cpu_allocator),
      cpu_allocator);

  GenericTensorAccessorW result = 
    tensor_accessor_elementwise_divide(numerator, denominator, cpu_allocator);

  if (attrs.elementwise_affine) {
    GenericTensorAccessorR resolved_gamma = assert_unwrap(gamma);
    GenericTensorAccessorR resolved_beta = assert_unwrap(beta);

    TensorDims promoted_gamma_dims = 
      tensor_dims_transform_with_idx(
        input.shape.dims,
        [&](ff_dim_t dim_idx, positive_int dim_size) -> positive_int {
          if (contains(attrs.axes, dim_idx)) {
            return dim_size;
          } else {
            return 1_p;
          }
        });

    TensorDims promoted_beta_dims = promoted_gamma_dims;

    GenericTensorAccessorR promoted_gamma = 
      tensor_accessor_promote_dims(resolved_gamma, promoted_gamma_dims);

    GenericTensorAccessorR promoted_beta = 
      tensor_accessor_promote_dims(resolved_beta, promoted_beta_dims);

    GenericTensorAccessorW broadcasted_gamma = broadcast(promoted_gamma);
    GenericTensorAccessorW broadcasted_beta = broadcast(promoted_beta);

    tensor_accessor_elementwise_add_to(
      tensor_accessor_elementwise_multiply(
        result,
        broadcasted_gamma,
        cpu_allocator),
      broadcasted_beta,
      output);
  } else {
    copy_accessor_data_to_l_from_r(output, result);
  }
}

void layer_norm_cpu_backward_kernel(LayerNormAttrs const &,
                                    GenericTensorAccessorR const &output_grad,
                                    GenericTensorAccessorR const &input,
                                    GenericTensorAccessorW const &input_grad,
                                    GenericTensorAccessorR const &gamma,
                                    GenericTensorAccessorW const &gamma_grad,
                                    GenericTensorAccessorW const &beta_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
