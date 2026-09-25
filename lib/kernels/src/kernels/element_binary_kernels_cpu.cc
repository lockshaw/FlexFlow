#include "kernels/element_binary_kernels_cpu.h"
#include "utils/not_implemented.h"
#include "kernels/tensor_accessor_binary_ops.h"

namespace FlexFlow {

void element_binary_cpu_forward_kernel(
  ElementBinaryAttrs const &attrs,
  GenericTensorAccessorR const &lhs,
  GenericTensorAccessorR const &rhs,
  GenericTensorAccessorW const &out)
{
  switch (attrs.op) {
    case ElementBinaryOp::ADD:
    {
      tensor_accessor_elementwise_add_to(lhs, rhs, out);
      break;
    }
    case ElementBinaryOp::SUBTRACT:
    {
      tensor_accessor_elementwise_subtract_to(lhs, rhs, out);
      break;
    }
    case ElementBinaryOp::MULTIPLY:
    {
      tensor_accessor_elementwise_multiply_to(lhs, rhs, out);
      break;
    }
    case ElementBinaryOp::DIVIDE:
    {
      tensor_accessor_elementwise_divide_to(lhs, rhs, out);
      break;
    }
    case ElementBinaryOp::MAX:
    {
      tensor_accessor_elementwise_max_to(lhs, rhs, out);
      break;
    }
    case ElementBinaryOp::MIN:
    {
      tensor_accessor_elementwise_min_to(lhs, rhs, out);
      break;
    }
    default:
      PANIC("Unknown ElementBinaryOp {}", attrs.op);
  }
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
