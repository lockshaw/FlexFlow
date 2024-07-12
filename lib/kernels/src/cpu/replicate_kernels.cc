#include "kernels/datatype_dispatch.h"
#include "kernels/replicate_kernels_cpu.h"

namespace FlexFlow {
namespace Kernels {
namespace Replicate {
namespace CPU {

template <typename T>
void replicate_backward_kernel(T *input,
                               T const *output,
                               size_t num_elements,
                               size_t num_replicas) {
  for (size_t i = 0; i < num_elements; ++i) {
    T sum = 0;
    for (size_t j = 0; j < num_replicas; ++j) {
      sum += output[j * num_elements + i];
    }
    input[i] = sum;
  }
}

// Why does replicate forward seem to only transfer memory? Shouldn't it also
// handle the replication?
template <DataType T>
struct ForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    memcpy(output.get<T>(),
           input.get<T>(),
           input.shape.num_elements() * size_of_datatype(T));
  }
};

template <DataType T>
struct BackwardKernel {
  void operator()(GenericTensorAccessorW const &input,
                  GenericTensorAccessorR const &output,
                  size_t num_replicas) {
    size_t total_elements = input.shape.num_elements() * num_replicas;
    replicate_backward_kernel(
        input.get<T>(), output.get<T>(), total_elements, num_replicas);
  }
};

void forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(input.data_type, input, output);
}

void backward_kernel(GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output,
                     size_t num_replicas) {
  DataTypeDispatch1<BackwardKernel>{}(
      input.data_type, input, output, num_replicas);
}

} // namespace CPU
} // namespace Replicate
} // namespace Kernels
} // namespace FlexFlow
