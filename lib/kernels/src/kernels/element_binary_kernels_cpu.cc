#include "kernels/element_binary_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void element_binary_cpu_forward_kernel(
  ElementBinaryAttrs const &attrs,
  GenericTensorAccessorR const &lhs,
  GenericTensorAccessorR const &rhs,
  GenericTensorAccessorR const &out)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

void element_binary_cpu_backward_kernel(
  ElementBinaryAttrs const &attrs,
  GenericTensorAccessorR const &lhs,
  GenericTensorAccessorW const &lhs_grad,
  GenericTensorAccessorR const &rhs,
  GenericTensorAccessorW const &rhs_grad,
  GenericTensorAccessorR const &output,
  GenericTensorAccessorR const &output_grad)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
