#include "kernels/softmax_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void softmax_cpu_forward_kernel(SoftmaxAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  GenericTensorAccessorW exponentiated = 
    tensor_accessor_exp(input, cpu_allocator);

  GenericTensorAccessorW denominator =  
    reduce_tensor_accessor_in_dims(
      exponentiated,
      std::set<ff_dim_t>{attrs.dim},
      cpu_allocator,
      [&](float accum, float x) -> float {
        return accum + x;
      });

  GenericTensorAccessorW broadcasted_denominator = 
    tensor_accessor_broadcast(
      denominator,
      input.shape.dims,
      cpu_allocator);

  return tensor_accessor_elementwise_divide_to(
    exponentiated,
    broadcasted_denominator,
    output);
}

void softmax_cpu_backward_kernel(SoftmaxAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
