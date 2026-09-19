#include "kernels/split_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void split_cpu_forward_kernel(SplitAttrs const &attrs,
                              GenericTensorAccessorR const &input,
                              std::vector<GenericTensorAccessorW> const &outputs)
{
  tensor_accessor_split_to(
    /*input=*/input, 
    /*axis=*/attrs.axis,
    /*sizes=*/attrs.sizes,
    /*outputs=*/outputs);
}

void split_cpu_backward_kernel(SplitAttrs const &attrs,
                               std::vector<GenericTensorAccessorR> const &output_grads,
                               GenericTensorAccessorW const &input_grad)
{
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  GenericTensorAccessorW result = foldl1(output_grads,
                [&](GenericTensorAccessorR const &accum, GenericTensorAccessorR const &elem) 
                  -> GenericTensorAccessorR
                {
                  return tensor_accessor_binary_concat(accum, elem, attrs.axis, cpu_allocator);
                });
  
  copy_accessor_data_to_l_from_r(
    /*dst_accessor=*/input_grad,
    /*src_accessor=*/result);
}

} // namespace FlexFlow
