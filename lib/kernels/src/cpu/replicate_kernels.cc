#include "kernels/datatype_dispatch.h"
#include "kernels/replicate_kernels_cpu.h"

namespace FlexFlow::Kernels::Replicate {

template <DataType DT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output) {
    memcpy(output.get<DT>(),
           input.get<DT>(),
           get_volume(input.shape).unwrap_nonnegative() *
               size_of_datatype(DT).unwrap_nonnegative());
  }
};

template <DataType DT>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &output,
                  GenericTensorAccessorW &input,
                  size_t num_elements,
                  size_t num_replicas) {
    using T = real_type_t<DT>;
    for (int i = 0; i < num_elements; i++) {
      T cur_sum = 0;
      for (int j = 0; j < num_replicas; j++) {
        cur_sum += output.at<DT>({i, j});
      }
      input.at<DT>(nonnegative_int{i}) = cur_sum;
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW &output) {
  DataTypeDispatch1<CPUForwardKernel>{}(input.data_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW &input,
                         size_t num_replicas) {
  size_t num_elements = get_volume(input.shape).unwrap_nonnegative();
  DataTypeDispatch1<CPUBackwardKernel>{}(
      input.data_type, output, input, num_elements, num_replicas);
}

} // namespace FlexFlow::Kernels::Replicate
