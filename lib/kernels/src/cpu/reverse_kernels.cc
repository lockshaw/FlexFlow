#include "kernels/datatype_dispatch.h"
#include "kernels/reverse_kernels_cpu.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <vector>

namespace FlexFlow::Kernels::Reverse {

template <DataType DT>
struct CPUReverseForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  int num_out_blks,
                  int reverse_dim_size,
                  int in_blk_size) {
    for (nonnegative_int blk_idx : nonnegative_range(nonnegative_int{num_out_blks})) {
      for (nonnegative_int rev_idx : nonnegative_range(nonnegative_range{reverse_dim_size})) {
        for (nonnegative_int inner_idx : nonnegative_range(nonnegative_range{in_blk_size})) {
          nonnegative_int reversed_idx = nonnegative_int{
            reverse_dim_size - 1 - rev_idx.unwrap_nonnegative()
          };

          output.at<DT>({inner_idx, rev_idx, blk_idx}) = input.at<DT>(
              {inner_idx, reversed_idx, blk_idx});
        }
      }
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor,
                        int num_out_blks,
                        int reverse_dim_size,
                        int in_blk_size) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(input_accessor.data_type,
                                               input_accessor,
                                               output_accessor,
                                               num_out_blks,
                                               reverse_dim_size,
                                               in_blk_size);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_accessor,
                         GenericTensorAccessorW &input_accessor,
                         int num_out_blks,
                         int reverse_dim_size,
                         int in_blk_size) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(output_accessor.data_type,
                                               output_accessor,
                                               input_accessor,
                                               num_out_blks,
                                               reverse_dim_size,
                                               in_blk_size);
}

} // namespace FlexFlow::Kernels::Reverse
