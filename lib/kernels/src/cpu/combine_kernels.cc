#include "kernels/combine_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow::Kernels::Combine {

template <DataType DT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    memcpy(output.get<DT>(),
           input.get<DT>(),
           (get_volume(input.shape) * size_of_datatype(DT)).unwrap_nonnegative());
  }
};

template <DataType DT>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) {
    for (int i = 0; i < get_volume(output_grad.shape); ++i) {
      input_grad.get<DT>()[i] += output_grad.get<DT>()[i];
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  DataTypeDispatch1<CPUForwardKernel>{}(input.data_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                         GenericTensorAccessorW const &input_grad) {
  DataTypeDispatch1<CPUBackwardKernel>{}(
      input_grad.data_type, output_grad, input_grad);
}

} // namespace FlexFlow::Kernels::Combine
